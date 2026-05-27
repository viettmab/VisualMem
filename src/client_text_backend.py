import json
import os
import time

import requests
from dotenv import load_dotenv


load_dotenv()

class MemosApiOnlineClient:
    def __init__(self):
        self.memos_url = os.getenv("MEMOS_ONLINE_URL")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": os.getenv("MEMOS_KEY"),
        }

    def add(self, messages, user_id, conv_id=None, batch_size: int = 9999):
        url = f"{self.memos_url}/add/message"
        for i in range(0, len(messages), batch_size):
            batch_messages = messages[i : i + batch_size]
            payload = json.dumps(
                {
                    "messages": batch_messages,
                    "user_id": user_id,
                    "conversation_id": conv_id,
                }
            )

            max_retries = 5
            for attempt in range(max_retries):
                try:
                    response = requests.request("POST", url, data=payload, headers=self.headers)
                    assert response.status_code == 200, response.text
                    assert json.loads(response.text)["message"] == "ok", response.text
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(30)
                    else:
                        raise e

    def search(self, query, user_id, top_k):
        url = f"{self.memos_url}/search/memory"
        payload = json.dumps(
            {
                "query": query,
                "user_id": user_id,
                "memory_limit_number": top_k,
                "mode": os.getenv("SEARCH_MODE", "fast"),
                "include_preference": True,
                "pref_top_k": 6,
            }
        )

        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = requests.request("POST", url, data=payload, headers=self.headers)
                assert response.status_code == 200, response.text
                assert json.loads(response.text)["message"] == "ok", response.text
                data = json.loads(response.text)["data"]
                text_mem_res = data["memory_detail_list"]
                pref_mem_res = data["preference_detail_list"]
                preference_note = data["preference_note"]

                for item in text_mem_res:
                    item.update({"memory": item.pop("memory_value")})

                explicit_pref_string = "Explicit Preference:"
                implicit_pref_string = "\n\nImplicit Preference:"
                explicit_idx = 0
                implicit_idx = 0
                for pref in pref_mem_res:
                    if pref["preference_type"] == "explicit_preference":
                        explicit_pref_string += f"\n{explicit_idx + 1}. {pref['preference']}"
                        explicit_idx += 1
                    if pref["preference_type"] == "implicit_preference":
                        implicit_pref_string += f"\n{implicit_idx + 1}. {pref['preference']}"
                        implicit_idx += 1

                return {
                    "text_mem": [{"memories": text_mem_res}],
                    "pref_string": explicit_pref_string + implicit_pref_string + preference_note,
                }

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    raise e