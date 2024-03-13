def set_anritsu(f):
    """NB make sure Anritsu server is running"""
    import requests
    r = requests.get(f'http://skynet.physics.ucsb.edu:51111/loset/{f}')
    return r.json()
