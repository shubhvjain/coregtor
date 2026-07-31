class CoRegTorError(Exception):
    """CoregTor error with error code and details."""
    def __init__(self, message: str, code: str = "GENERAL", details: dict = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}

