import os


def update_urdf_mass(directory, old_mass_value="1.0", new_mass_value="0.5"):
    # Traverse through all the files in the directory
    for filename in os.listdir(directory):
        # Check if the file is a URDF file
        if filename.endswith(".urdf"):
            file_path = os.path.join(directory, filename)

            # Read the content of the URDF file
            with open(file_path, 'r') as file:
                file_content = file.read()

            # Replace the old mass value with the new mass value
            updated_content = file_content.replace(f'<mass value="{old_mass_value}" />',
                                                   f'<mass value="{new_mass_value}" />')

            # Write the updated content back to the URDF file
            with open(file_path, 'w') as file:
                file.write(updated_content)

    print(f"All URDF files in {directory} have been updated.")


if __name__ == "__main__":
    # Usage example
    directory_path = '/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/object2urdf/ws/egad_train'  # Replace this with the actual directory containing the URDF files
    update_urdf_mass(directory_path)
