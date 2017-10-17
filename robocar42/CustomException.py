
class ValidationError(Exception):
	def __init__(self, message):
		super(ValidationError, self).__init__(message)

class DisplayError(Exception):
    def __init__(self, mesage):
        super(DisplayError, self).__init__(message)
