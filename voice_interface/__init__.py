def __getattr__(name):
    if name == "VoiceCommandInterface":
        from .core import VoiceCommandInterface
        return VoiceCommandInterface
    raise AttributeError(name)
