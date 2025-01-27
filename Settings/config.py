import yaml


def getPaths(key):
    with open("/Users/luisafaust/Desktop/LFA_PTSFC_GIT/Settings/inputPath.yaml", "r") as file:
        data = yaml.safe_load(file)
    return data.get("paths", {}).get(key, "key not found")