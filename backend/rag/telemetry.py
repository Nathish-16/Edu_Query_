import os
from dotenv import load_dotenv

# Load environmental configurations
# Look in current working directory first
load_dotenv()
# Also search in the absolute path of the backend directory as a robust fallback
backend_env = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
if os.path.exists(backend_env):
    load_dotenv(backend_env)

observe = None
langfuse_context = None
is_tracing_active = False

try:
    # Check for Langfuse SDK
    # In newer SDK versions (v3/v4), observe is imported directly from langfuse.
    # In legacy v2 versions, it is imported from langfuse.decorators along with langfuse_context.
    try:
        from langfuse import observe as lf_observe
        lf_context = None
    except ImportError:
        from langfuse.decorators import observe as lf_observe, langfuse_context as lf_context
    
    # Verify environment keys are configured
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    
    if public_key and secret_key:
        print("[TELEMETRY] Langfuse environment variables resolved. Observability tracing active.")
        observe = lf_observe
        langfuse_context = lf_context
        is_tracing_active = True
    else:
        # Langfuse installed but keys missing. Define silent mock decorators.
        print("[TELEMETRY] Langfuse keys missing in .env. Tracing running in silent mock mode.")
        
        def mock_observe(*args, **kwargs):
            if len(args) == 1 and callable(args[0]):
                return args[0]
            def decorator(func):
                return func
            return decorator
            
        class MockContext:
            def update_current_trace(self, *args, **kwargs): pass
            def update_current_observation(self, *args, **kwargs): pass
            
        observe = mock_observe
        langfuse_context = MockContext()
        is_tracing_active = False
        
except ImportError:
    # Langfuse SDK not installed. Fallback to pass-through decorations.
    print("[TELEMETRY] Langfuse package not installed. Running in silent mock mode.")
    
    def mock_observe(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def decorator(func):
            return func
        return decorator
        
    class MockContext:
        def update_current_trace(self, *args, **kwargs): pass
        def update_current_observation(self, *args, **kwargs): pass
        
    observe = mock_observe
    langfuse_context = MockContext()
    is_tracing_active = False

def get_telemetry_status():
    return {
        "tracing_active": is_tracing_active,
        "provider": "langfuse" if is_tracing_active else "none"
    }
