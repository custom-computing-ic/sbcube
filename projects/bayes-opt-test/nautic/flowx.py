from functools import wraps

# ✅ Reuse or redefine config structure
class FlowxConfig:
    disable_nautic = False

flowx_cfg = FlowxConfig()

# Cache the flow import (initially None)
_cached_flow = None

# cached import prefect
def __get_flow():
    global _cached_flow
    if _cached_flow is None:
        from prefect import flow
        _cached_flow = flow
    return _cached_flow

def flowx(_func=None, **f_kwargs):
    def decorator(fn):
        fn._is_flowx = True  # 👈 optional marker

        @wraps(fn)
        def wrapper(ctx, *args, **kwargs):
            fn(ctx, *args, **kwargs)
            return ctx
        if flowx_cfg.disable_nautic:
            return wrapper  # Just a plain callable
        else:
            __flow = __get_flow()  # ✅ Import and cache if needed
            return __flow(**f_kwargs)(wrapper)

    if _func is None:
        return decorator
    else:
        return decorator(_func)
