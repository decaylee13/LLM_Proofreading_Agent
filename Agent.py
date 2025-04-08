from ChromeNGL import ChromeNGL
from utils.utils import parse_action
import time
import json
from utils.Values import Values
from utils.utils import find_mouse_increments, find_pos_increments, find_crossSectionScale_increments, find_projectionOrientation_increments, find_projectionScale_increments
from utils.maths import quaternion_to_euler, euler_to_quaternion
import copy
import os
import argparse
import PIL

class Agent:
    def __init__(self, headless=False, start_session: bool = False, verbose:bool =False):
        """
            Agent class that controls the ChromeNGL instance.
            Main use cases are to make automated actions in the Neuroglancer viewer.
            The agent can be used to follow an episode (JSON recording), or to take actions in the viewer.
            Args:
                headless: Boolean to start the Neuroglancer viewer in headless mode.
                start_session: Boolean to start the Neuroglancer viewer session.
                debug: Boolean to print debug information.

            The idea is to have the Agent asking a model the appropriate action to take and then applying it by handing it the ChromeNGL instance it owns.
        """
        self.values = Values()
        self.verbose = verbose
        if start_session:
            self.chrome_ngl = ChromeNGL(headless=headless, values=self.values, verbose=self.verbose)
            self.chrome_ngl.start_session()
        else:
            print("ChromeNGL instance not started, can't use the agent to interact with the viewer")
            self.chrome_ngl = None

        """Initializing the mouse position to the center of the image, in a RL setting you could track and move this mouse position."""
        self.mouse_x = self.values.data_image_width // 2
        self.mouse_y = self.values.data_image_height // 2


    def start_neuroglancer_session(self, localHost:bool =False)->None:
        """
        Starts a Neuroglancer session.
        Args:
            localHost: Whether to use the local host session. Elsewise uses the neuroglancer demo session.
        """
        if self.chrome_ngl is None:
            raise ValueError("ChromeNGL instance not started, can't use the agent to interact with the viewer")
        self.chrome_ngl.start_neuroglancer_session(localHost=localHost)

    def start_graphene_session(self, localHost:bool =False)->None:
        """
        Starts a Graphene session. This requires passing the middle-auth login which is hard to automate. Be very careful when using this and adapt accordingly.
        Args:
            localHost: Whether to use the local host session. Elsewise uses the graphene demo session.
        """
        if self.chrome_ngl is None:
            raise ValueError("ChromeNGL instance not started, can't use the agent to interact with the viewer")
        self.chrome_ngl.start_graphene_session(localHost=localHost)

    def reset_mouse(self)->None:
        """
        Resets the mouse position to the center of the image.
        """
        self.mouse_x = self.values.data_image_width // 2
        self.mouse_y = self.values.data_image_height // 2

    def prepare_state(self, image_path:str =None, euler_angles:bool =False,resize:bool =False, add_mouse:bool =False, fast:bool =True)->tuple[list, PIL.Image.Image, dict]:
        """
            Core function. Returns a list for the state and the current image as a PIL image (default from ChromeNGL class)
            Args:
                image_path: Path to save the image. If not specified, the image is not saved.
                euler_angles: Boolean to return the state in euler angles.
                resize: Boolean to resize the image.
                add_mouse: Boolean to add the mouse position to the image.

            Returns:
                pos_state: List of position, crossSectionScale, projectionOrientation, projectionScale
                curr_image: PIL image of the current state
                json_state: JSON state of the current state as a dictionary
        """
        state = self.chrome_ngl.get_JSON_state()
        json_state = json.loads(state)
        position = json_state["position"]
        crossSectionScale = json_state["crossSectionScale"]
        if "projectionOrientation" in json_state:
            projectionOrientation = json_state["projectionOrientation"]
        else:
            default_orientation = [0, 0, 0, 1]
            projectionOrientation = default_orientation
            json_state["projectionOrientation"] = default_orientation
        projectionScale = json_state["projectionScale"]
        if add_mouse:
            curr_image = self.chrome_ngl.get_screenshot(image_path, resize=resize, mouse_x=self.mouse_x, mouse_y=self.mouse_y, fast=fast)
        else:
            curr_image = self.chrome_ngl.get_screenshot(image_path, resize=resize, fast=fast)
        if euler_angles:
            projectionOrientationEuler = quaternion_to_euler(projectionOrientation)
            pos_state = [position, crossSectionScale, projectionOrientationEuler, projectionScale]
        else:
            pos_state = [position, crossSectionScale, projectionOrientation, projectionScale]
        return pos_state, curr_image, json_state

    def get_state(self, euler_angles:bool =False)->list:
        """
            Get the current state of the Neuroglancer viewer. Skips the Image retrieval.
            Args:
                euler_angles: Boolean to return the state in euler angles instead of quaternions.
        """
        state = self.chrome_ngl.get_JSON_state()
        json_state = json.loads(state)
        position = json_state["position"]
        crossSectionScale = json_state["crossSectionScale"]
        projectionOrientation = json_state["projectionOrientation"]
        projectionScale = json_state["projectionScale"]
        if euler_angles:
            projectionOrientationEuler = quaternion_to_euler(projectionOrientation)
            pos_state = [position, crossSectionScale, projectionOrientationEuler, projectionScale]
        else:
            pos_state = [position, crossSectionScale, projectionOrientation, projectionScale]
        return pos_state


    def mouse_position(self)->tuple[int, int]:
        return self.mouse_x, self.mouse_y
        
    def apply_actions(self, output_vector:list, euler_angles:bool =False,json_state:dict =None, verbose:bool =False)->None:
        """
            Takes an output_vector of the ActorCritic (discrete actions argmaxed) and transforms it to an environment action that is handled by ChromeNGL
        """
        if euler_angles:
            (
            left_click, right_click, double_click,  # 3 booleans
            x, y,                                  # 2 floats for mouse position
            key_Shift, key_Ctrl, key_Alt,          # 3 booleans for keys
            json_change,                           # 1 boolean for JSON change
            delta_position_x, delta_position_y, delta_position_z,  # 3 floats
            delta_crossSectionScale,               # 1 float
            delta_projectionOrientation_e1, delta_projectionOrientation_e2, delta_projectionOrientation_e3,  # 3 floats
            delta_projectionScale                  # 1 float
            ) = [v for v in output_vector] # before [v.item() if isinstance(v, torch.Tensor) else v for v in output_vector]
        else:
            (
            left_click, right_click, double_click,  # 3 booleans
            x, y,                                  # 2 floats for mouse position
            key_Shift, key_Ctrl, key_Alt,          # 3 booleans for keys
            json_change,                           # 1 boolean for JSON change
            delta_position_x, delta_position_y, delta_position_z,  # 3 floats
            delta_crossSectionScale,               # 1 float
            delta_projectionOrientation_q1, delta_projectionOrientation_q2,
            delta_projectionOrientation_q3, delta_projectionOrientation_q4,  # 4 floats
            delta_projectionScale                  # 1 float
            ) = [v for v in output_vector] # before [v.item() if isinstance(v, torch.Tensor) else v for v in output_vector]
    
        if json_state is None:
            json_state = self.chrome_ngl.get_JSON_state()
            json_state = json.loads(json_state)

        # fitting output_vector back into action space
        x = abs(x) * self.values.x_factor_programmatic
        y = abs(y) * self.values.y_factor_programmatic
        key_pressed = ""
        if key_Shift:
            if verbose:
                print("Shift key pressed")
            key_pressed += "Shift, "
        if key_Ctrl:
            if verbose:
                print("Ctrl key pressed")
            key_pressed += "Ctrl, "
        if key_Alt:
            if verbose:
                print("Alt key pressed")
            key_pressed += "Alt, "
        key_pressed = key_pressed.strip(", ")

        if left_click:
            if verbose:
                print("Decided to do a left click at position", x, y)
            self.chrome_ngl.mouse_key_action(x, y, "left_click", key_pressed)
        elif right_click:
            if verbose:
                print("Decided to do a right click at position", x, y)
            self.chrome_ngl.mouse_key_action(x, y, "right_click", key_pressed)
        elif double_click:
            if verbose:
                print("Decided to do a double click at position", x, y)
            self.chrome_ngl.mouse_key_action(x, y, "double_click", key_pressed)
        elif json_change:
            if verbose:
                print("Decided to change the JSON state")
            old_position = json_state["position"][:]

            json_state["position"][0] += delta_position_x
            json_state["position"][1] += delta_position_y
            json_state["position"][2] += delta_position_z
            if verbose:
                print(f"Position updated: {old_position} -> {json_state['position']}")

            old_crossSectionScale = json_state["crossSectionScale"]
            # crossSectionScale is a multiplicative factor calculated on the previous value: coeff = (new_value - old_value) / old_value
            json_state["crossSectionScale"] += delta_crossSectionScale
            if verbose:
                print(f"CrossSectionScale updated: {old_crossSectionScale:.6f} -> {json_state['crossSectionScale']:.6f}")

            if euler_angles:
                old_projectionOrientation = quaternion_to_euler(json_state["projectionOrientation"])
                new_projectionOrientation = [
                    old_projectionOrientation[0] + delta_projectionOrientation_e1,
                    old_projectionOrientation[1] + delta_projectionOrientation_e2,
                    old_projectionOrientation[2] + delta_projectionOrientation_e3
                ]
                json_state["projectionOrientation"] = euler_to_quaternion(new_projectionOrientation)
            else:    
                old_projectionOrientation = json_state["projectionOrientation"][:]
                json_state["projectionOrientation"][0] += delta_projectionOrientation_q1
                json_state["projectionOrientation"][1] += delta_projectionOrientation_q2
                json_state["projectionOrientation"][2] += delta_projectionOrientation_q3 
                json_state["projectionOrientation"][3] += delta_projectionOrientation_q4
            if verbose:
                print(f"ProjectionOrientation updated: {old_projectionOrientation} -> {json_state['projectionOrientation']}")

            old_projectionScale = json_state["projectionScale"]
            json_state["projectionScale"] = min(500000, json_state["projectionScale"] + delta_projectionScale)
            if verbose:
                print(f"ProjectionScale updated: {old_projectionScale:.6f} -> {json_state['projectionScale']:.6f}")
            self.chrome_ngl.change_JSON_state_url(json_state)
        if verbose:
            print("Decision acted upon")

    def apply_actions_QN(self, chosen_action: int, grid=False, exact_pos=(None,None)):
        """
        The action chosen is an index in the Q action space. This function re-translates
        the action back into the environment and applies the necessary updates.
        """
        if grid:
            action_name = self.values.grid_action_translation_dict[chosen_action]
            mouse_action_limit = len(self.values.grid_action_translation_dict) - 1
        else:
            action_name = self.values.grid_action_translation_dict[chosen_action]
            mouse_action_limit = self.values.click_limit_index

        if action_name is None:
            print("Invalid action index")
            return
        if self.verbose:
            print(f"Chosen action: {action_name}")    

        if action_name.startswith("move_to_box"):
            x, y = action_name.split("_")[-2:]
            x, y = int(x), int(y)
            #print(f"1. Moving to box at position ({x}, {y})")
            #self.mouse_x = x + self.values.grid_size_x // 2
            #self.mouse_y = y + self.values.grid_size_y // 2
            if exact_pos != (None, None):
                mouse_x, mouse_y = exact_pos
                mouse_y -= self.values.neuroglancer_margin_top # returning the mouse position into the coordinate system used in selenium
                self.chrome_ngl.mouse_key_action(mouse_x, mouse_y, "right_click")
            else:
                self.mouse_x = x + self.values.grid_size_x // 2
                self.mouse_y = y + self.values.grid_size_y // 2
                self.mouse_y -= self.values.neuroglancer_margin_top # returning the mouse position into the coordinate system used in selenium
                if self.mouse_x <= 900:
                    self.mouse_y = max(40, self.mouse_y) # to avoid neuroglancer's top bar that toggles view
                try:
                    self.chrome_ngl.mouse_key_action(self.mouse_x, self.mouse_y, "right_click")
                except:
                    print("The mouse action failed")

        elif chosen_action <= mouse_action_limit:
            #mouse_x_factor, mouse_y_factor = self.values.delta_mouse_x_factor, self.values.delta_mouse_y_factor
            mouse_x, mouse_y = self.mouse_position()
            # Apply the chosen action
            if action_name == "Left click":
                print(f"Performing left click at position ({mouse_x}, {mouse_y})")
                mouse_y -= self.values.neuroglancer_margin_top # returning the mouse position into the coordinate system used in selenium
                self.chrome_ngl.mouse_key_action(mouse_x, mouse_y, "left_click")

            elif action_name == "Right click":
                print(f"Performing right click at position ({mouse_x}, {mouse_y})")
                mouse_y -= self.values.neuroglancer_margin_top # returning the mouse position into the coordinate system used in selenium
                self.chrome_ngl.mouse_key_action(mouse_x, mouse_y, "right_click")

            elif action_name == "Double click":
                print(f"Performing double click at position ({mouse_x}, {mouse_y})")
                mouse_y -= self.values.neuroglancer_margin_top # returning the mouse position into the coordinate system used in selenium
                self.chrome_ngl.mouse_key_action(mouse_x, mouse_y, "double_click")

            elif action_name.startswith("incr_mouse_x"):
                increment = int(action_name.split("_")[-1])
                self.mouse_x = min(self.mouse_x + increment, self.values.data_image_width)

            elif action_name.startswith("decr_mouse_x"):
                decrement = int(action_name.split("_")[-1])
                self.mouse_x = max(0,self.mouse_x - decrement)

            elif action_name.startswith("incr_mouse_y"):
                increment = int(action_name.split("_")[-1])
                self.mouse_y = min(self.mouse_y + increment, self.values.data_image_height)

            elif action_name.startswith("decr_mouse_y"):
                decrement = int(action_name.split("_")[-1])
                self.mouse_y = max(0, self.mouse_y - decrement)
            print("Applied: ", action_name)
        else:
            # Retrieve the current JSON state
            json_state = self.chrome_ngl.get_JSON_state()
            json_state = json.loads(json_state)
            if action_name.startswith("incr_") or action_name.startswith("decr_"):
                action_parts = action_name.split("_")
                action_type = action_parts[1]  # e.g., position, crossSectionScale, etc.
                adjustment = float(action_parts[-1])  # Extract adjustment value from suffix
                is_increase = action_name.startswith("incr_")

                if action_type == "position":
                    axis = {"x": 0, "y": 1, "z": 2}[action_parts[2]]  # Map x, y, z to indices
                    current_value = json_state["position"][axis]
                    limit = self.values.max_position if is_increase else self.values.min_position
                    json_state["position"][axis] = (
                        min(current_value + adjustment, limit[axis])
                        if is_increase else max(current_value - adjustment, limit[axis])
                    )

                elif action_type == "crossSectionScale":
                    current_value = json_state["crossSectionScale"]
                    limit = self.values.max_crossSectionScale if is_increase else self.values.min_crossSectionScale
                    json_state["crossSectionScale"] = (
                        min(current_value + adjustment, limit)
                        if is_increase else max(current_value - adjustment, limit)
                    )
                elif action_type == "projectionOrientation":
                    # the projectionOrientation from the json state is in quaternion form.
                    # The model outputs a change in euler angles. We need to conver the json state to euler angles. Update the euler angles and convert back to quaternion
                    index = int(action_parts[3][-1]) - 1  # Extract quaternion index (q1, q2, etc.)
                    current_value_quat = json_state["projectionOrientation"]
                    current_value_euler = quaternion_to_euler(current_value_quat)
                    current_value_euler[index] += adjustment if is_increase else -adjustment
                    json_state["projectionOrientation"] = euler_to_quaternion(current_value_euler)
                    #json_state["projectionOrientation"][index] += adjustment if is_increase else -adjustment

                elif action_type == "projectionScale":
                    current_value = json_state["projectionScale"]
                    limit = self.values.max_projectionScale if is_increase else self.values.min_projectionScale
                    json_state["projectionScale"] = (
                        min(current_value + adjustment, limit)
                        if is_increase else max(current_value - adjustment, limit)
                    )
            self.chrome_ngl.change_JSON_state_url(json_state)

    def follow_episode(self, episode_path:str, sleep_time:float =0.1)->None:
        """"
        This function takes a recording (JSON episode) and follows the actions of the user in the Neuroglancer viewer step by step
        Action clicks are handled by the MouseActionHandler and JSON changes are handled by the change_JSON_state_url method.
        """

        episode = json.load(open(episode_path))
        self.chrome_ngl.change_JSON_state_url(json.dumps(episode[0]["state"]))

        for i in range(1,len(episode)):
            step = episode[i] # state_step is a dictionary containing keys: state, action, time
            step_state = step["state"]
            step_action = step["action"]
            step_time = step["time"]
            parsed_action, direct_json_change = parse_action(step_action)
            print("Step: ", i, "Action: ", step_action)
            if direct_json_change:
                json_state = json.dumps(step_state)
                self.chrome_ngl.change_JSON_state_url(json_state)
            else:
                self.chrome_ngl.mouse_key_action(parsed_action['x'], parsed_action['y'], parsed_action['click_type'], parsed_action['keys_pressed'])
            time.sleep(sleep_time)

    def reset(self)->None:
        self.chrome_ngl.start_neuroglancer_session()
        self.reset_mouse()

    def parse_episode(self, episode_path:str, save_path:str =None, wait:bool =False, add_mouse:bool =False)->list:
        """
        # Function for parsing the episode data into a format that can be used for pretraining (imitation learning)
        # To call this function, we need to start the session first. Then it will change states and take screenshots
        # Currently returns in the output vector format with quaternions for orientation.
        """
        os.makedirs(save_path, exist_ok=True)
        episode = json.load(open(episode_path))
        parsed_data = []
        print("---------Parsing episode---------")
        print("--Save path: ", save_path, "--Episode path: ", episode_path, "--Add mouse: ", add_mouse, "--Wait: ", wait)
        print("--Number of steps: ", len(episode))

        for i in range(0, len(episode)-1):
            self.chrome_ngl.change_JSON_state_url(json.dumps(episode[i]["state"]))
            if wait:
                time.sleep(0.2)
            # we build the action that leads from the previous state to the current state
            # We need to be careful here, the recording saves the action that led to the state with it, not the action taken in the state
            next_episode = episode[i+1]
            current_episode = episode[i]
            next_state = next_episode["state"]
            current_state = current_episode["state"]
            next_action = next_episode["action"]
            output_vector = [
                0, 0, 0,  # left_click, right_click, double_click
                0.0, 0.0,  # x, y (mouse position)
                0, 0, 0,  # key_Shift, key_Ctrl, key_Alt
                0,  # json_change
                0.0, 0.0, 0.0,  # delta_position_x, delta_position_y, delta_position_z
                0.0,  # delta_crossSectionScale
                0.0, 0.0, 0.0, 0.0,  # delta_projectionOrientation_q1, q2, q3, q4
                0.0  # delta_projectionScale
            ]
            screen_action=False
            if "Double Click" in next_action:
                output_vector[2] = 1
                screen_action=True   
            elif "Left Click" in next_action:
                output_vector[0] = 1  
                screen_action=True
            elif "Right Click" in next_action:
                output_vector[1] = 1  
                screen_action=True

            if "Relative position" in next_action:
                position_data = next_action.split("Relative position: ")[1].split(" with keys:")[0]
                position_parts = position_data.split(", y=")
                x = int(position_parts[0].replace("x=", "").strip())
                y = int(float(position_parts[1].strip())) + self.values.neuroglancer_margin_top # adding the margin, now all positions are relative to the neuroglancer window
                
                output_vector[3] = x
                output_vector[4] = y
            
            if "Shift" in next_action:
                output_vector[5] = 1 
            if "Ctrl" in next_action:
                output_vector[6] = 1 
            if "Alt" in next_action:
                output_vector[7] = 1 
            
            if screen_action == False:
                # This means the action was a JSON change
                output_vector[8] = 1
            
                delta_pos = [
                    (next_state["position"][0] - current_state["position"][0]),
                    (next_state["position"][1] - current_state["position"][1]),
                    (next_state["position"][2] - current_state["position"][2])
                ]
                output_vector[9], output_vector[10], output_vector[11] = delta_pos
                
                output_vector[12] = next_state["crossSectionScale"]-current_state["crossSectionScale"]
                
                delta_orientation = [
                    (next_state["projectionOrientation"][0] - current_state["projectionOrientation"][0]),
                    (next_state["projectionOrientation"][1] - current_state["projectionOrientation"][1]),
                    (next_state["projectionOrientation"][2] - current_state["projectionOrientation"][2]),
                    (next_state["projectionOrientation"][3] - current_state["projectionOrientation"][3])
                ]
                output_vector[13], output_vector[14], output_vector[15], output_vector[16] = delta_orientation
                
                output_vector[17] = next_state["projectionScale"] - current_state["projectionScale"]
            
            pos_state, curr_image, json_state = self.prepare_state(image_path=f"{save_path}/screenshots/" + str(i) + ".png", add_mouse=add_mouse, fast=True)

            parsed_data.append({
                "pos_state": pos_state,  # original state for reference
                "action_vector": tuple(output_vector),  # encoded action vector
                "action": next_action,  # action description (optional)
                "json_state": json_state  # the current JSON state (optional)
            })
        with open(f"{save_path}/data.json", "w") as f:
            json.dump(parsed_data, f, indent=4)
        print("---------Episode parsed---------")
        return parsed_data
    
    def parse_episode_for_QN(self, episode, save_path=None, add_mouse=True, grid=True):
        # We are simply going to change the output vector to be in the QN action space
        # For that, we need to discretize every change to be a succession of possible Q actions.
        parsed_data = []
        #curr_mouse_x = curr_mouse_y = step = 0
        self.reset_mouse()
        step = 0
        
        self.chrome_ngl.refresh()
        time.sleep(1)

        def process_increments(incr_list, positions, key):
            nonlocal step
            for i, incr in enumerate(incr_list):
                print("Changing JSON: Incr: ", incr, "Position: ", positions[i], " Key is ", key, " Step is ", step)
                #q_output_vector = [0] * self.values.num_Q_actions
                #q_output_vector[incr] = 1
                pos_state_euler, curr_image, json_state = self.prepare_state(
                    image_path=f"{save_path}/screenshots/{step}.png", euler_angles=True, add_mouse=add_mouse
                )
                pos_state = pos_state_euler + [self.mouse_x, self.mouse_y]
                parsed_data.append({
                    "pos_state": pos_state,
                    "action_index": incr,
                    "action_name": key
                })
                step += 1

                # Applying action that will lead to the next state
                intermediate_state = json_state
                
                if key == "projectionOrientation":
                    # Change the quaternion to euler angles for neuroglancer understanding of orientation
                    intermediate_state[key] = euler_to_quaternion(positions[i])
                else:
                    intermediate_state[key] = positions[i]
                self.chrome_ngl.change_JSON_state_url(json.dumps(intermediate_state))
                time.sleep(0.2)
        i =0
        while i < len(episode) - 1:
            current_episode = episode[i]
            # Find the next non-relative position state
            j = i + 1
            while j < len(episode)-1 and "Relative position" not in episode[j]["action"] and "Relative position" not in episode[j+1]["action"]:
                print("Skipping: ", episode[j]["action"])
                # stops when j+1 is a mouse click
                j += 1
            
            if j >= len(episode):
                break
            next_episode = episode[j]  
            next_action = next_episode["action"]
            print("Next action: ", next_action)
    
            self.chrome_ngl.change_JSON_state_url(json.dumps(current_episode["state"]))
            time.sleep(1)

            if "Relative position" not in next_action:
                # This means a JSON change action. 
                # The idea here is to find the increments to reach the next state. Since the variables are continuous, we need to discretize them
                # We also use the fact they are independent to find the increments for each variable separately

                current_state = copy.deepcopy(current_episode["state"])
                next_state= copy.deepcopy(next_episode["state"])
                
                current_state["projectionOrientation"] = quaternion_to_euler(current_state["projectionOrientation"])
                next_state["projectionOrientation"] = quaternion_to_euler(next_state["projectionOrientation"])
                
                for key, find_fn in [
                    ("position", find_pos_increments),
                    ("crossSectionScale", find_crossSectionScale_increments),
                    ("projectionOrientation", find_projectionOrientation_increments),
                    ("projectionScale", find_projectionScale_increments)
                ]:
                    incr_list, positions = find_fn(current_state[key], next_state[key], self.values, grid=grid)
                    process_increments(incr_list, positions, key)    
            else:    
                position_data = next_action.split("Relative position: ")[1].split(" with keys:")[0]
                position_parts = position_data.split(", y=")
                x = int(position_parts[0].replace("x=", "").strip())
                y = int(float(position_parts[1].strip()))

                objective_x = x
                objective_y = y
                
                if grid:
                    # We immediately go the closest box
                    pos_state, curr_image, json_state = self.prepare_state(
                        image_path=f"{save_path}/screenshots/" + str(step) + ".png", euler_angles=True, add_mouse=add_mouse
                    )
                    self.mouse_x, self.mouse_y = self.values.map_to_grid(objective_x, objective_y)
                    incr = self.values.Q_action_space_indexes_grid[f"move_to_box_{self.mouse_x}_{self.mouse_y}"]
                    self.mouse_x += self.values.grid_size_x // 2
                    self.mouse_y += self.values.grid_size_y // 2
                    
                    pos_state = pos_state + [self.mouse_x, self.mouse_y]
                    parsed_data.append({
                        "pos_state": pos_state,
                        "action_index": incr,
                        "click_pos": (objective_x, objective_y),
                        "action_name": f"move_to_box_{self.mouse_x}_{self.mouse_y}"
                    })
                    step += 1
                if "Double Click" in next_action:
                    #q_output_vector[self.values.Q_action_space_indexes["Double click"]] = 1
                    incr = self.values.Q_action_space_indexes["Double click"]
                elif "Left Click" in next_action:
                    incr = self.values.Q_action_space_indexes["Left click"]
                elif "Right Click" in next_action:
                    incr = self.values.Q_action_space_indexes["Right click"]
                pos_state, curr_image, json_state = self.prepare_state(
                    image_path=f"{save_path}/screenshots/" + str(step) + ".png", euler_angles=True, add_mouse=add_mouse
                )
                pos_state = pos_state + [self.mouse_x, self.mouse_y]
                parsed_data.append({
                        "pos_state": pos_state,
                        "action_index": incr,
                        "action_name": next_action
                    })
                step += 1
            i = j

        return parsed_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start',type=int, help="Bottom range of episodes to parse")
    parser.add_argument('--end',type=int, help="Upper range of episodes to parse")
    parser.add_argument('--headless', action='store_true', help="Run the agent in headless mode")
    parser.add_argument('--add_mouse', action='store_true', help="Add mouse to the screenshots")
    parser.add_argument('--grid', action='store_true', help="Parse the mouse clicks to the grid")
    args = parser.parse_args()

    rl_agent = Agent(start_session=True, headless=args.headless, debug=False)
    rl_agent.chrome_ngl.start_neuroglancer_session()
    
    # set url to correct start
    #rl_agent.chrome_ngl.change_url("http://localhost:8000/client/#!%7B%22dimensions%22:%7B%22x%22:%5B4e-9%2C%22m%22%5D%2C%22y%22:%5B4e-9%2C%22m%22%5D%2C%22z%22:%5B4e-8%2C%22m%22%5D%7D%2C%22position%22:%5B130037.875%2C77002.2109375%2C1809.5506591796875%5D%2C%22crossSectionScale%22:1.8496565995583267%2C%22projectionOrientation%22:%5B0.7071067690849304%2C0%2C-0.7071067690849304%2C0%5D%2C%22projectionScale%22:17155.89749730396%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%22precomputed://https://bossdb-open-data.s3.amazonaws.com/flywire/fafbv14%22%2C%22tab%22:%22source%22%2C%22name%22:%22Maryland%20%28USA%29-image%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%22precomputed://gs://flywire_v141_m783%22%2C%22tab%22:%22segments%22%2C%22segments%22:%5B%22%21720575940623044103%22%2C%22720575940620096171%22%5D%2C%22name%22:%22flywire_v141_m783%22%7D%5D%2C%22showDefaultAnnotations%22:false%2C%22selectedLayer%22:%7B%22size%22:350%2C%22visible%22:true%2C%22layer%22:%22flywire_v141_m783%22%7D%2C%22layout%22:%22xy-3d%22%7D")
    # time.sleep(3)
    # print("Session started")
    # while True:
    #     # ask for input in apply_actions_QN
    #     user_input = input("Enter action index: ").strip()
    #     if user_input == "exit":
    #         break
    #     rl_agent.apply_actions_QN(int(user_input))
    
    for i in range(args.start, args.end+1):
        file_path = f"./episodes/1800x900/episode_{i}.json"
        save_path = f"./reparsed_episodes/test/episode_{i}"
        print(f"Processing episode {i}")
        parsed_data = rl_agent.parse_episode(file_path, save_path, add_mouse=args.add_mouse)




    # for i in range(2,3):
    #     file_path = f"./episodes/1800x900/episode_{i}.json"
    #     data = json.load(open(file_path, "r"))
    #     rl_agent.follow_episode(data)


    

        