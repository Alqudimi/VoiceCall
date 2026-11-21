"""Iraqi Voice Studio — production-grade RVC v2 voice conversion platform.

Layered architecture (see docs/ARCHITECTURE.md):

    core          domain entities, interfaces (Protocols), errors, events
    application   use-case services (conversion / training / realtime / tunnel)
    engine        RVC v2 model engine (vendored nets + our adapters)
    training      data pipeline + GAN trainer with resumable checkpoints
    realtime      low-latency streaming conversion engine
    infrastructure config, logging, audio DSP, persistence, processes, security
    api           FastAPI backend + static web SPA
    cli           unified command line interface (mirrors the Makefile)
"""

__version__ = "1.4.1"
APP_NAME = "Iraqi Voice Studio"
APP_SLUG = "rvc-platform"
