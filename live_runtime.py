"""Process-lifetime singletons for the live feedback lane.

Streamlit re-executes app.py top to bottom on every rerun, and the recorder
makes that happen roughly twice a second during a lesson. Anything bound at
module level in app.py is therefore rebuilt at that rate - which silently
defeated a "shared" client cache and churned a new thread pool per rerun.

Imports are cached in sys.modules, so this module's body runs once per process.
State that must outlive a rerun belongs here.
"""

import concurrent.futures
import threading

# Its own pool: BACKGROUND_EXECUTOR has two workers shared with Replicate and
# post-lesson synthesis, and a long Whisper run would block every live call.
EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=3, thread_name_prefix="live-feedback"
)

_CLIENTS = {}
_CLIENTS_LOCK = threading.Lock()


def openai_client(api_key, base_url, timeout):
    """One client per key, reused for the life of the process.

    A client per call meant a fresh TLS handshake every time, which is pure
    latency on a path the teacher is watching.
    """
    key = (api_key, base_url, timeout)
    with _CLIENTS_LOCK:
        client = _CLIENTS.get(key)
        if client is None:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=base_url,
                            timeout=timeout, max_retries=0)
            _CLIENTS[key] = client
        return client
