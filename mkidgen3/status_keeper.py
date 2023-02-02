# import redis

class StatusKeeper:
    def __init__(self, location):
        """

        Args:
            location: location of a redis server for keeping status info according to some sort of schema
        """
        self._location = location

        self._redis = None #Redis(location)

    def update(self, namespace, **data):
        pass
