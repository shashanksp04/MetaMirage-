import pysqlite3
import sys
sys.modules["sqlite3"] = pysqlite3
sys.path.append('../')
from chat_models.OpenAI_Chat import OpenAI_Chat
from chat_models.Client import Client
from pydantic import BaseModel
import json
import multiprocessing
import os
from tqdm import tqdm
import argparse
import time

# # NEW CODE - Add debug logging to inspect API requests
# import logging
# import httpx

# # Enable debug logging to see what's being sent to vLLM
# logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')

# # Monkey-patch httpx to log all API requests and responses
# _original_httpx_post = httpx.AsyncClient.post

# async def _logged_httpx_post(self, *args, **kwargs):
#     """Log HTTP POST requests to see tool schemas being sent"""
#     url = args[0] if args else kwargs.get('url', 'unknown')
#     print(f"\n{'='*80}")
#     print(f"[HTTP DEBUG] POST to: {url}")
    
#     # Log request body if it's JSON
#     if 'json' in kwargs:
#         json_body = kwargs['json']
#         print(f"[HTTP DEBUG] Request JSON keys: {list(json_body.keys())}")
        
#         # Log tools if present
#         if 'tools' in json_body:
#             tools = json_body['tools']
#             if tools:
#                 print(f"[HTTP DEBUG] ✓ Tools present in request: {len(tools)} tool(s)")
#                 for i, tool in enumerate(tools[:3]):  # Show first 3 tools
#                     tool_name = tool.get('function', {}).get('name', 'unknown')
#                     print(f"[HTTP DEBUG]   Tool {i+1}: {tool_name}")
#                 if len(tools) > 3:
#                     print(f"[HTTP DEBUG]   ... and {len(tools) - 3} more")
#             else:
#                 print(f"[HTTP DEBUG] ✗ WARNING: 'tools' key present but empty!")
#         else:
#             print(f"[HTTP DEBUG] ✗ WARNING: No 'tools' key in request!")
        
#         # Log messages
#         if 'messages' in json_body:
#             messages = json_body['messages']
#             print(f"[HTTP DEBUG] Messages: {len(messages)} message(s)")
    
#     # Make the actual request
#     response = await _original_httpx_post(self, *args, **kwargs)
    
#     # Log response
#     print(f"[HTTP DEBUG] Response status: {response.status_code}")
#     try:
#         resp_json = response.json()
#         if 'choices' in resp_json:
#             choice = resp_json['choices'][0]
#             message = choice.get('message', {})
            
#             # Check for tool calls in response
#             if 'tool_calls' in message and message['tool_calls']:
#                 print(f"[HTTP DEBUG] ✓ Response contains {len(message['tool_calls'])} tool call(s)")
#                 for tc in message['tool_calls'][:3]:
#                     print(f"[HTTP DEBUG]   - {tc.get('function', {}).get('name', 'unknown')}")
#             else:
#                 # Check if response is just text
#                 content = message.get('content', '')
#                 if content:
#                     print(f"[HTTP DEBUG] ✗ Response is text only (no tool calls): {content[:100]}...")
#                 else:
#                     print(f"[HTTP DEBUG] ✗ Response has no content or tool calls")
#     except Exception as e:
#         print(f"[HTTP DEBUG] Could not parse response JSON: {e}")
    
#     print(f"{'='*80}\n")
#     return response

# # Apply the monkey patch
# httpx.AsyncClient.post = _logged_httpx_post
# # END NEW CODE

# Global RAG worker function for multiprocessing
def rag_worker_process(rag_queue, result_dict, test_model, embed_model_name, device, api_base):
    """Separate process that handles all RAG requests to avoid multiple model loads"""
    import asyncio
    from rag_agent.main import MainAgent
    try:
        # Use default port 11434 if api_base is empty
        if not api_base or api_base == "":
            api_base = "http://127.0.0.1:11434/v1"
        rag_agent = MainAgent(test_model=test_model, embed_model_name=embed_model_name, device=device, api_base=api_base)
        rag_agent.reset_collection()
        rag_runner = rag_agent.main()
        
        # Create event loop for this process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        request_count = 0
        RESTART_INTERVAL = 1000
        
        while True:
            request = rag_queue.get()
            if request is None:  # Poison pill to stop worker
                break
            
            item_id, query = request

            request_count += 1
            if request_count % RESTART_INTERVAL == 0:
                print(f"[RAG Worker] Item {item_id}: ⟳ Recreating agent after {request_count} requests...")
                try:
                    rag_agent = MainAgent(test_model=test_model, embed_model_name=embed_model_name, device=device, api_base=api_base)
                    rag_runner = rag_agent.main()
                    print(f"[RAG Worker] Item {item_id}: ✓ Agent recreated successfully")
                except Exception as e:
                    print(f"[RAG Worker] Item {item_id}: ✗ Failed to recreate agent: {e}")

            print(f"[RAG Worker] Processing item {item_id}: Received query (length: {len(query)} chars)")
            try:
                print(f"[RAG Worker] Item {item_id}: Starting RAG agent processing...")
                # Debug: Check if agent has tools registered
                if hasattr(rag_runner, 'agent'):
                    try:
                        # Check tools attribute (the list passed to LlmAgent)
                        if hasattr(rag_runner.agent, 'tools'):
                            tools_attr = rag_runner.agent.tools
                            if tools_attr is not None:
                                tool_count = len(tools_attr) if isinstance(tools_attr, (list, tuple)) else 0
                                print(f"[RAG Worker] Item {item_id}: Agent has {tool_count} tool(s) in tools attribute")
                            else:
                                print(f"[RAG Worker] Item {item_id}: Agent tools attribute is None")
                        else:
                            print(f"[RAG Worker] Item {item_id}: Agent does not have 'tools' attribute")
                    except Exception as e:
                        print(f"[RAG Worker] Item {item_id}: Could not check tool count: {e}")
                
                # run_debug is async, so we need to await it
                # Try to use a unique session ID per query to avoid session state issues
                # If session_id parameter doesn't exist, fall back to default behavior
                try:
                    session_id = f"rag_session_{item_id}"
                    # session_id = "rag_session_shared"
                    rag_response = loop.run_until_complete(rag_runner.run_debug(query, session_id=session_id))
                    print(f"[RAG Worker] Item {item_id}: Used session_id={session_id}")
                except TypeError:
                    # If session_id parameter doesn't exist, use default behavior (same session)
                    rag_response = loop.run_until_complete(rag_runner.run_debug(query))
                    print(f"[RAG Worker] Item {item_id}: Using default session (session_id not supported)")
                print(f"[RAG Worker] Item {item_id}: RAG agent completed.")
                print(f"[RAG Worker] Item {item_id}: Response type: {type(rag_response)}")
                
                # Extract agent's answer from the response
                # run_debug returns a list of Event objects, we need to extract text from them
                rag_answer = None
                web_search_performed = False  # Initialize web_search_performed flag
                print(f"[RAG Worker] Item {item_id}: Starting response extraction...")
                
                if rag_response is None:
                    print(f"[RAG Worker] Item {item_id}: WARNING - rag_response is None!")
                elif isinstance(rag_response, list):
                    print(f"[RAG Worker] Item {item_id}: Response is a list with {len(rag_response)} events")
                    # Debug: Check all events to see if tools were called
                    tool_calls_found = []
                    agent_texts = []
                    for i, event in enumerate(rag_response):
                        event_type = type(event).__name__
                        author = getattr(event, 'author', 'unknown')
                        print(f"[RAG Worker] Item {item_id}: Event {i}: type={event_type}, author={author}")
                        
                        # Check for tool calls in the event
                        function_calls = event.get_function_calls()
                        if function_calls:
                            print(f"[RAG Worker] Item {item_id}:   → Found {len(function_calls)} tool call(s) in event {i}")
                            for tc in function_calls:
                                tool_name = getattr(tc, 'name', 'unknown')
                                tool_calls_found.append(tool_name)
                                print(f"[RAG Worker] Item {item_id}:     - Tool: {tool_name}")
                        
                        # Check if this is from the Rag_Agent
                        if author == 'Rag_Agent':
                            # Extract text from content.parts
                            if hasattr(event, 'content') and hasattr(event.content, 'parts'):
                                for part in event.content.parts:
                                    if hasattr(part, 'text') and part.text:
                                        agent_texts.append(part.text)
                    
                    if tool_calls_found:
                        print(f"[RAG Worker] Item {item_id}: ✓ Found tool calls: {', '.join(tool_calls_found)}")
                    else:
                        print(f"[RAG Worker] Item {item_id}: ✗ WARNING - No tool calls found in any events!")
                    
                    # Check if web_search was performed
                    web_search_performed = 'web_search' in tool_calls_found or '_tracked_web_search' in tool_calls_found
                    if web_search_performed:
                        print(f"[RAG Worker] Item {item_id}: ✓ Web search was performed")
                    
                    if agent_texts:
                        # Get the last agent response (most recent)
                        rag_answer = agent_texts[-1].strip()
                        print(f"[RAG Worker] Item {item_id}: ✓ Extracted agent text from Event objects, length: {len(rag_answer)} chars")
                    else:
                        print(f"[RAG Worker] Item {item_id}: WARNING - No agent text found in events!")
                else:
                    # Fallback: try to convert to string and extract using regex
                    rag_response_str = str(rag_response)
                    print(f"[RAG Worker] Item {item_id}: Response is not a list, trying string extraction...")
                    # Check if web_search appears in string representation (fallback detection)
                    if 'web_search' in rag_response_str.lower() or '_tracked_web_search' in rag_response_str:
                        web_search_performed = True
                        print(f"[RAG Worker] Item {item_id}: ✓ Web search detected in string representation")
                    import re
                    # Try to extract text from the string representation (look for text="""...""")
                    text_match = re.search(r'text="""(.*?)"""', rag_response_str, re.DOTALL)
                    if text_match:
                        rag_answer = text_match.group(1).strip()
                        print(f"[RAG Worker] Item {item_id}: ✓ Extracted text from string representation, length: {len(rag_answer)} chars")
                
                # Accept answers that are at least 5 characters (reduced from 10 to handle short responses)
                if rag_answer and len(rag_answer) >= 5:
                    # Check if the response is just a list of tool names (common failure mode)
                    tool_names = ['_tracked_retrieve_content', '_tracked_evaluate_confidence', 
                    '_tracked_web_search', '_tracked_add_web_content', '_tracked_add_pdf_content', 
                    '_tracked_extract_keywords']

                    rag_answer_lower = rag_answer.lower()
                    # Count how many tool names appear in the response
                    tool_name_count = sum(1 for tool_name in tool_names if tool_name.lower() in rag_answer_lower)
                    
                    # If the response contains mostly tool names and nothing else, it's likely a failure
                    if tool_name_count >= 1 and len(rag_answer.split('\n')) <= tool_name_count + 2:
                        print(f"[RAG Worker] Item {item_id}: ✗ FAILED - Response is just tool names, not actual results")
                        print(f"[RAG Worker] Item {item_id}:   Response: {repr(rag_answer[:100])}")
                        result_dict[item_id] = (None, "Agent returned tool names instead of calling tools or returning results", False)
                    else:
                        print(f"[RAG Worker] Item {item_id}: ✓ SUCCESS - Extracted valid answer ({len(rag_answer)} chars)")
                        result_dict[item_id] = (rag_answer, None, web_search_performed)
                else:
                    print(f"[RAG Worker] Item {item_id}: ✗ FAILED - Answer extraction failed")
                    print(f"[RAG Worker] Item {item_id}:   - rag_answer is None: {rag_answer is None}")
                    print(f"[RAG Worker] Item {item_id}:   - rag_answer length: {len(rag_answer) if rag_answer else 0}")
                    # Debug: print what we found to help diagnose
                    if rag_response_str:
                        has_rag_agent = "Rag_Agent" in rag_response_str
                        if has_rag_agent:
                            # Print a snippet of the response to see what's happening
                            rag_pos = rag_response_str.rfind("Rag_Agent")
                            snippet = rag_response_str[max(0, rag_pos-50):min(len(rag_response_str), rag_pos+200)]
                            print(f"[RAG Worker] Item {item_id}: DEBUG - Found Rag_Agent at pos {rag_pos}")
                            print(f"[RAG Worker] Item {item_id}: DEBUG - Response snippet: {repr(snippet)}")
                    result_dict[item_id] = (None, "No RAG answer found in response", False)
            except Exception as e:
                print(f"[RAG Worker] Item {item_id}: ✗ EXCEPTION - Error during processing: {str(e)}")
                import traceback
                print(f"[RAG Worker] Item {item_id}: Traceback:\n{traceback.format_exc()}")
                result_dict[item_id] = (None, str(e), False)
    except Exception as e:
        # If initialization fails, mark all pending requests with error
        print(f"RAG worker initialization failed: {e}")
        while True:
            try:
                request = rag_queue.get_nowait()
                if request is None:
                    break
                item_id, _ = request
                result_dict[item_id] = (None, f"RAG worker initialization failed: {str(e)}", False)
            except:
                break

class Generate:
    def __init__(self, raw_data_file, output_file, model_name="gpt-4o", openai_api_base="", num_processes=None, 
    embed_model_name="BAAI/bge-base-en-v1.5", 
    test_model="Qwen2.5-VL-3B-Instruct",
    device="None"):
        self.raw_data_file = raw_data_file
        self.output_file = output_file
        self.offline_model = model_name
        self.model_name = model_name.split("/")[-1]
        self.openai_api_base = openai_api_base
        # If the number of processes is not specified, use the number of CPU cores
        self.num_processes = num_processes if num_processes is not None else os.cpu_count()
        # Store RAG config instead of initializing here (to avoid GPU memory issues)
        self.test_model = test_model
        self.embed_model_name = embed_model_name
        self.device = device

    def get_prompt(self, item):
        question = item["question"]
        user_prompt = f"{question}"
        images = item.get("images", [])
        new_images = []
        for i in range(len(images)):
            dir_path = os.path.dirname(os.path.abspath(self.raw_data_file))  
            new_path = dir_path + "/" + images[i]
            if not os.path.exists(new_path):
                print(f"Image path {new_path} does not exist. Please check the input data.")
                continue
            new_images.append(new_path)
        return {"user": user_prompt, "images": new_images}

    def safe_qsize(self, q):
        try:
            return q.qsize()
        except (NotImplementedError, AttributeError):
            return 0
        except Exception:
            return 0


    # Function to handle item processing
    def process_item(self, args):
        item, model_name, output_file, lock, rag_queue, rag_result_dict = args
        prompt = self.get_prompt(item)
        response = None
        last_exception = None
        self.max_retries = 5
        self.retry_delay = 5  # seconds
        item_id = item.get('id', 'unknown')
        
        # Request RAG response via queue
        enhanced_query = prompt["user"]  # Default to original query
        rag_implementation = False  # Will be True if RAG succeeds
        rag_status = None  # Will contain "successful" or error message
        if rag_queue is not None:
            qsize = self.safe_qsize(rag_queue)
            rag_queue.put((item_id, prompt["user"]))
            # Wait for RAG response (with timeout)
            base_timeout = 60
            extra = 5 * max(0, qsize)  # 5s per queued request (tune)
            max_wait_time = base_timeout + extra
            wait_interval = 0.1  # seconds
            waited = 0
            while item_id not in rag_result_dict and waited < max_wait_time:
                time.sleep(wait_interval)
                waited += wait_interval
            
            if item_id in rag_result_dict:
                rag_result = rag_result_dict[item_id]
                # Handle both old format (rag_answer, rag_error) and new format (rag_answer, rag_error, web_search_performed)
                if len(rag_result) == 3:
                    rag_answer, rag_error, web_search_performed = rag_result
                else:
                    # Old format without web_search_performed flag
                    rag_answer, rag_error = rag_result
                    web_search_performed = False
                
                if rag_answer and rag_error is None:
                    print(f"[Main Process] Item {item_id}: ✓ RAG SUCCESS - Answer received ({len(rag_answer)} chars)")
                    enhanced_query = f"{prompt['user']}\n\nadditional context: {rag_answer}"
                    rag_implementation = True
                    rag_status = "successful"
                    item["RAG_web_search_performed"] = web_search_performed
                    if web_search_performed:
                        print(f"[Main Process] Item {item_id}: ✓ Web search was performed during RAG")
                else:
                    print(f"[Main Process] Item {item_id}: ✗ RAG FAILED - Error: {rag_error}")
                    print(f"[Main Process] Item {item_id}:   - rag_answer is None: {rag_answer is None}")
                    print(f"[Main Process] Item {item_id}:   - rag_answer length: {len(rag_answer) if rag_answer else 0}")
                    print(f"[Main Process] Item {item_id}: Using original query (no RAG enhancement)")
                    rag_implementation = False
                    rag_status = rag_error if rag_error else "No RAG answer found in response"
                    item["RAG_web_search_performed"] = False
            else:
                print(f"[Main Process] Item {item_id}: ✗ RAG TIMEOUT - No response after {max_wait_time}s")
                print(f"[Main Process] Item {item_id}: Using original query (no RAG enhancement)")
                rag_implementation = False
                rag_status = "timeout"
                item["RAG_web_search_performed"] = False
        else:
            # RAG queue is None (RAG disabled)
            rag_implementation = False
            rag_status = "rag_disabled"
            item["RAG_web_search_performed"] = False
        
        if not rag_implementation:
            item["RAG_implementation"] = False
            item["RAG_status"] = rag_status
            # Clean up RAG result from shared dict if it exists
            if rag_result_dict is not None and item_id in rag_result_dict:
                del rag_result_dict[item_id]
            # Write item without model response and return early
            with lock:
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            return item.get('id')
        else:
            for attempt in range(self.max_retries):
                try:
                    # Initialize the client based on the model name
                    if self.model_name.startswith("gpt"):
                        client = OpenAI_Chat(model_name=model_name, messages=[])
                    else:
                        client = Client(model_name=self.offline_model, openai_api_base=self.openai_api_base, messages=[])
                    
                    response = client.chat(prompt=enhanced_query, images=prompt["images"])
                    item[model_name] = response
                    # item["info"] = client.info() # Uncomment if needed
                    item["history"] = client.get_history()
                    break # Exit retry loop on success

                except Exception as e:
                    last_exception = e # Store the exception
                    print(f"Attempt {attempt + 1}/{self.max_retries} failed for item {item_id}: {e}")
                    if attempt < self.max_retries - 1:
                        
                        print(f"Waiting {self.retry_delay} seconds before retrying...")
                        time.sleep(self.retry_delay)
                    else:
                        # Max retries reached
                        print(f"Max retries ({self.max_retries}) reached for item {item_id}. Marking as failed.")
                        item[model_name] = -1 # Mark as failed after all retries
            
            # Clean up RAG result from shared dict
            if rag_result_dict is not None and item_id in rag_result_dict:
                del rag_result_dict[item_id]
            
            item["RAG_implementation"] = True
            item["RAG_status"] = "successful"
 
        with lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        return item.get('id')

    def generate(self):
        # Read the raw data file
        with open(self.raw_data_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Check if the output file exists and read processed items
        processed_ids = set()
        if os.path.exists(self.output_file):
            with open(self.output_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        if self.model_name in item and item[self.model_name] != -1 and item[self.model_name] is not None:
                            processed_ids.add(item["id"])
                    except json.JSONDecodeError:
                        # Handle potentially corrupt JSON lines
                        continue

        total_items = len(data)
        already_processed = len(processed_ids)
        items_to_process = [item for item in data if item.get("id") not in processed_ids]

        if already_processed > 0:
            print(
                f"Resuming processing: {already_processed} items already completed, "
                f"{len(items_to_process)} items remaining."
            )
        else:
            print(f"Starting fresh: Processing {len(items_to_process)} items.")

        if items_to_process:
            # IMPORTANT: use spawn context to avoid CUDA + fork issues
            ctx = multiprocessing.get_context("spawn")

            manager = ctx.Manager()
            lock = manager.Lock()
            rag_queue = manager.Queue()
            rag_result_dict = manager.dict()  # Shared dict to store RAG results

            # Start RAG worker process (single instance to save GPU memory)
            rag_process = ctx.Process(
                target=rag_worker_process,
                args=(
                    rag_queue,
                    rag_result_dict,
                    self.test_model,
                    self.embed_model_name,
                    self.device,
                    self.openai_api_base,
                ),
            )
            rag_process.start()

            # Initialize the process pool with the specified number of processes
            pool = ctx.Pool(processes=self.num_processes)
            args_list = [
                (item, self.model_name, self.output_file, lock, rag_queue, rag_result_dict)
                for item in items_to_process
            ]

            # Use tqdm to show progress
            try:
                for _ in tqdm(
                    pool.imap_unordered(self.process_item, args_list),
                    total=len(args_list),
                    desc="Processing items",
                ):
                    pass
            finally:
                # Signal RAG worker to stop
                try:
                    rag_queue.put(None)
                except Exception:
                    pass

                # Cleanly close the pool
                try:
                    pool.close()
                except Exception:
                    pass
                try:
                    pool.join()
                except Exception:
                    pass

                # Wait for RAG worker to finish
                rag_process.join(timeout=30)
                if rag_process.is_alive():
                    print("RAG worker did not terminate gracefully, forcing termination...")
                    rag_process.terminate()
                    rag_process.join()

        print("Processing completed.")
        print(f"Summary: {already_processed} items processed, {len(items_to_process)} items remaining.")
        self.cleanup_output(len(data))


    def cleanup_output(self, data_length):
        valid_items = []
        
        with open(self.output_file, "r", encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if self.model_name in item and item[self.model_name] != -1 and item[self.model_name] != None:
                        valid_items.append(item)
                except json.JSONDecodeError:
                    continue

        with open(self.output_file, "w", encoding='utf-8') as f:
            for item in valid_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"Total successful items: {len(valid_items)}. \n Remaining items to process: {data_length - len(valid_items)}.")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description="Generate responses using LLMs model.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="Model name to use.")
    parser.add_argument("--openai_api_base", type=str, default="", help="Base URL for OpenAI API.")
    parser.add_argument("--num_processes", type=int, default=os.cpu_count(), help="Number of processes to use.")
    parser.add_argument("--embed_model_name", type=str, default="BAAI/bge-base-en-v1.5", help="Embedding model name to use.")
    parser.add_argument("--test_model", type=str, default="Qwen2.5-VL-3B-Instruct", help="Test model name to use.")
    parser.add_argument("--device", type=str, default="None", help="Device to use.")
    args = parser.parse_args()

    reformatter = Generate(raw_data_file=args.input_file, output_file=args.output_file, model_name=args.model_name, num_processes=args.num_processes, openai_api_base=args.openai_api_base, embed_model_name=args.embed_model_name, test_model=args.test_model, device=args.device)
    reformatter.generate()
