import requests
import json

API_KEY = ""
LABEL_STUDIO_URL = "http://localhost:8080"

def create_project() -> None:
    """
    Creates a new project in Label Studio.

    This function sends a POST request to the Label Studio API endpoint
    `/api/projects` with the provided title and description. It handles
    the response, checks for errors, and returns the project ID and headers.

    Args:
        None

    Returns:
        A tuple containing the project ID (string) and the request headers
        (dictionary).  If an error occurs during the API call, it raises a
        requests.exceptions.HTTPError exception.
    """
    headers = {
        "Authorization": f"Token {API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "title": "Materials",
        "description": "Material annotation project",
    }
    response = requests.post(
        f"{LABEL_STUDIO_URL}/api/projects", headers=headers, data=json.dumps(data)
    )

    response.raise_for_status()
    project = response.json()
    project_id = project["id"]
    print(f"Project created with ID: {project_id}")
    return project_id, headers

def import_data(headers: dict, project_id: str) -> str | None:
    """
    Imports data into a Label Studio project.

    This function sends a POST request to the Label Studio API endpoint
    `/api/storages/localfiles` with the provided project ID and path. It handles
    the response, checks for errors, and returns the storage ID if successful.

    Args:
        headers (dict): The HTTP headers for the request, including authorization.
        project_id (str): The ID of the Label Studio project to import data into.

    Returns:
        str | None: The ID of the created storage object if the import was successful,
                   or None if an error occurred.
    """
    data = {
        "title":"Local Import",
        "description":"Local import",
        "project":f"{project_id}",
        "path":"/images",
        "use_blob_urls": "true"
    }
    response = requests.post(
        f"{LABEL_STUDIO_URL}/api/storages/localfiles",
        headers=headers,
        json=data,
    )

    try:
        response.raise_for_status()
        print("Label configuration updated successfully!")
        return response.json()["id"]
    except requests.exceptions.HTTPError as e:
        print(f"Error updating label configuration: {e}")
        print(response.text)  # Print the server's response
        return None
    
def validate(headers:dict, project_id:str, storage_id:str) -> bool | None:
    """
    Validates a Label Studio storage object.

    This function sends a POST request to the Label Studio API endpoint
    `/api/storages/localfiles/validate` with the provided project ID and storage ID.
    It handles the response, checks for errors, and returns True if validation is successful,
    False otherwise.

    Args:
        headers (dict): The HTTP headers for the request, including authorization.
        project_id (str): The ID of the Label Studio project.
        storage_id (str): The ID of the storage object to validate.

    Returns:
        bool | None: True if validation is successful, False otherwise.  If an error occurs,
                    it prints the error and response text and returns False.
    """
    data = {
        "project":f"{project_id}",
        "id":f"{storage_id}",
        "path":"/images"
    }

    response = requests.post(
        f"{LABEL_STUDIO_URL}/api/storages/localfiles/validate",
        headers=headers,
        json=data,
    )

    try:
        response.raise_for_status()
        print("Label configuration updated successfully!")
        return True
    except requests.exceptions.HTTPError as e:
        print(f"Error updating label configuration: {e}")
        print(response.text)  # Print the server's response
        return False

def sync_data(headers, project_id, storage_id):
    response = requests.post(
        f"{LABEL_STUDIO_URL}/api/storages/localfiles/{storage_id}/sync",
        headers=headers,
    )
    try:
        response.raise_for_status()
        print("Label configuration updated successfully!")
        return True
    except requests.exceptions.HTTPError as e:
        print(f"Error updating label configuration: {e}")
        print(response.text)  # Print the server's response
        return False

def create_config(headers:dict, project_id:str) -> bool | None:
    """
    Creates a Label Studio configuration XML for an image annotation task.

    This function sends a PATCH request to the Label Studio API endpoint
    `/api/projects/{project_id}` with the provided configuration XML. It handles
    the response, checks for errors, and returns True if successful, False otherwise.

    Args:
        headers (dict): The HTTP headers for the request, including authorization.
        project_id (str): The ID of the Label Studio project to update.

    Returns:
        bool | None: True if the configuration was updated successfully, False otherwise.
                    If an error occurs, it prints the error and response text and returns False.
    """
    
    label_config_xml = """
    <View>
      <Image name="image" value="$image"/>
      <RectangleLabels name="label" toName="image">
        <Label value="Betoni"/>
        <Label value="Teräs"/>
        <Label value="Muovi"/>
        <Label value="Materiaali ei tiedossa"/>
      </RectangleLabels>
      <Choices name="no_debris" toName="image">
        <Choice value="No material"/>
      </Choices>
    </View>
    """

    update_data = {"label_config": label_config_xml}
    response = requests.patch(
        f"{LABEL_STUDIO_URL}/api/projects/{project_id}",
        headers=headers,
        data=json.dumps(update_data),
    )

    try:
        response.raise_for_status()
        print("Label configuration updated successfully!")
        return True
    except requests.exceptions.HTTPError as e:
        print(f"Error updating label configuration: {e}")
        print(response.text)  # Print the server's response
        return False

if __name__=="__main__":
    API_KEY = input("Give your API key: ")
    project_id, headers = create_project()
    storage_id = import_data(headers,project_id)
    if storage_id:
        validation = validate(headers, project_id, storage_id)
    if validation:
        synced = sync_data(headers, project_id, storage_id)
    if synced:
        config = create_config(headers, project_id)
    if config:
        print("Project created succesfully")
        quit()



    


