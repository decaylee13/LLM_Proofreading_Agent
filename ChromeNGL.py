import platform
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from utils.MouseActionHandler import MouseActionHandler
import base64
import time
import json
import urllib.parse
from PIL import Image
import io
import os
from utils.Values import Values
from PIL import ImageDraw

class ChromeNGL:
    def __init__(self, headless:bool =False, values:Values =Values(), verbose:bool =False):
        """
        ChromeNGL handles the Chrome driver instance and handles all actions like screenshots, changing states through URLs, sending mouse actions to the MouseActionHandler.
        This class holds the lifetime of the Chrome instance.
        Args:
            headless: Opens Chrome without a GUI interface and reduces memory usage. Getting feedback in that case has to be done though get_screenshot / get_JSON_state.
            verbose: Prints verbose output to the console. Like when a session is started, when a screenshot is taken, when a JSON state is changed, etc.
        """
        
        self.values = values # core Values used. Change accordingly in the Values class.
        self.headless = headless
        self.verbose = verbose
        self.window_width = self.values.data_image_width
        self.window_height = self.values.data_image_height
        self.resize_width = self.values.model_image_width
        self.resize_height = self.values.model_image_height

        '''Login to Google Account'''
        self.mail_address = 'pnirlagent@gmail.com'
        self.password = 'secret-password'
        
        chrome_options = Options()
        if self.verbose:
            print("Headless mode:", headless)
        if self.headless:
            chrome_options.add_argument("--headless")
            #chrome_options.add_argument("--disable-gpu") provokes WebGL error
            chrome_options.add_argument("--enable-logging")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")

        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("useAutomationExtension", False)
        chrome_options.add_experimental_option("excludeSwitches",["enable-automation"])  

        if platform.system() == "Darwin":  # macOS
            chrome_service = Service("chromedrivers/chromedriver-mac-arm64/133/chromedriver")
            chrome_border_height = 87
            self.window_height += chrome_border_height
        elif platform.system() == "Windows":
            chrome_service = Service("K:/Coding Projects/Seung RL Agent/agent/NGL_Agent/chromedriver-win64/chromedriver.exe") #should be changed
            chrome_border_height = 95
            chrome_border_width = 16
            self.window_height += chrome_border_height
            self.window_width += chrome_border_width
            chrome_options.add_argument("--force-device-scale-factor=1")
        elif platform.system() == "Linux":  # Linux
            chrome_service = Service("/home/raphael/Documents/SAuto/chromedriver-linux64/chromedriver")
            chrome_options.binary_location = "/home/raphael/Documents/SAuto/chrome-linux64/google-chrome"
            chrome_border_height = 87 # Not tested if gives out proper 1800x900 output
            self.window_height += chrome_border_height

        chrome_options.add_argument(f"--window-size={self.window_width},{self.window_height}")   

        self.driver = webdriver.Chrome(service=chrome_service, options=chrome_options)
        self.init_url = 'https://accounts.google.com/Login'
        '''---------------------------------'''
	
        """Action space:"""
        self.action_handler = MouseActionHandler(self.driver)


        """Graphene session. Adding session that need middle-auth should be added similarly to the graphene session starter"""
        self.default_graphene_url = "http://localhost:8000/client/#!%7B%22dimensions%22%3A%20%7B%22x%22%3A%20%5B4e-09%2C%20%22m%22%5D%2C%20%22y%22%3A%20%5B4e-09%2C%20%22m%22%5D%2C%20%22z%22%3A%20%5B4e-08%2C%20%22m%22%5D%7D%2C%20%22layers%22%3A%20%5B%7B%22source%22%3A%20%22precomputed%3A//https%3A//bossdb-open-data.s3.amazonaws.com/flywire/fafbv14%22%2C%20%22type%22%3A%20%22image%22%2C%20%22tab%22%3A%20%22source%22%2C%20%22name%22%3A%20%22Maryland%20%28USA%29-image%22%7D%2C%20%7B%22tab%22%3A%20%22segments%22%2C%20%22source%22%3A%20%22graphene%3A//middleauth%2Bhttps%3A//prodv1.flywire-daf.com/segmentation/1.0/flywire_public%22%2C%20%22type%22%3A%20%22segmentation%22%2C%20%22segments%22%3A%20%5B%22720575940595846828%22%5D%2C%20%22colorSeed%22%3A%20883605311%2C%20%22name%22%3A%20%22Public%20Release%20783-segmentation%22%7D%5D%2C%20%22position%22%3A%20%5B174232.0%2C%2068224.0%2C%203870.0%5D%2C%20%22showDefaultAnnotations%22%3A%20false%2C%20%22perspectiveOrientation%22%3A%20%5B0%2C%200%2C%200%2C%201%5D%2C%20%22projectionScale%22%3A%201500%2C%20%22crossSectionScale%22%3A%200.5%2C%20%22jsonStateServer%22%3A%20%22https%3A//globalv1.flywire-daf.com/nglstate/post%22%2C%20%22selectedLayer%22%3A%20%7B%22layer%22%3A%20%22annotation%22%2C%20%22visible%22%3A%20true%7D%2C%20%22layout%22%3A%20%22xy-3d%22%7D"
  
    def end_session(self):
        self.driver.close()
        self.driver.quit()

    def start_session(self):
        self.driver.get(self.init_url)
        if self.verbose:
            print(f"Chrome session started. Navigated to URL: {self.init_url}")
        self.google_login()

    def start_neuroglancer_session(self, start_url: str = None, localHost:bool =False):
        """
        Starts a Neuroglancer session in the default Flywire session. This is the most common session used for training and does not require any additional login.
        Args:
            start_url: The URL to start the session on. If not specified, the default neuroglancer session will be used.
            localHost: Whether to use the local host session. Elsewise uses the neuroglancer demo session.

        For further url changing, use the change_url method or the change_JSON_state_url method.
        """
        if start_url is not None:
            self.change_url(start_url)
            return
        json_encoding = "#!%7B%22dimensions%22:%7B%22x%22:%5B4e-9%2C%22m%22%5D%2C%22y%22:%5B4e-9%2C%22m%22%5D%2C%22z%22:%5B4e-8%2C%22m%22%5D%7D%2C%22position%22:%5B143944.703125%2C61076.59375%2C192.5807647705078%5D%2C%22crossSectionScale%22:2.0339912586467497%2C%22projectionOrientation%22:%5B-0.4705163836479187%2C0.8044001460075378%2C-0.30343097448349%2C0.1987067461013794%5D%2C%22projectionScale%22:13976.00585680798%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%22precomputed://https://bossdb-open-data.s3.amazonaws.com/flywire/fafbv14%22%2C%22tab%22:%22source%22%2C%22name%22:%22Maryland%20%28USA%29-image%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%22precomputed://gs://flywire_v141_m783%22%2C%22tab%22:%22source%22%2C%22segments%22:%5B%22%21720575940623044103%22%2C%22%21720575940607208114%22%2C%22720575940603464672%22%5D%2C%22name%22:%22flywire_v141_m783%22%7D%5D%2C%22showDefaultAnnotations%22:false%2C%22selectedLayer%22:%7B%22size%22:350%2C%22layer%22:%22flywire_v141_m783%22%7D%2C%22layout%22:%22xy-3d%22%7D"
        neuroglancer_url = "https://neuroglancer-demo.appspot.com/"
        localHost_url = "http://localhost:8000/client/"
        time.sleep(1)
        if self.verbose:
            print("Starting Neuroglancer session...")
        if localHost:
            new_url = localHost_url + json_encoding
        else:
            new_url = neuroglancer_url + json_encoding
        if self.verbose:
            print(f"New URL: {new_url}")
        self.change_url(new_url)
        time.sleep(1)
        if self.verbose:
            print(f"Neuroglancer session started. Navigated to URL given.")

    def start_graphene_session(self, start_url: str = None):
        """This will start a local Host session using Graphene segmentation. This requires passing the middle-auth login which is hard to automate. Be very careful when using this and adapt accordingly.
        Args:
            start_url: The URL to start the session on. If not specified, the default graphene session will be used.
        Returns:
            None, session is started in place on the driver instance.
        """
        try:
            if start_url is None:
                start_url = self.default_graphene_url
            self.driver.get(start_url)
            wait = WebDriverWait(self.driver, 10)
            login_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//ul[@id='statusContainerModal']//button[text()='Login']")))
            login_button.click()
            main_window = self.driver.current_window_handle
            WebDriverWait(self.driver, 10).until(lambda d: len(d.window_handles) > 1)
            for handle in self.driver.window_handles:
                if handle != main_window:
                    self.driver.switch_to.window(handle)
                    break
            pni_rlagent = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//*[contains(text(), 'PNI RLAgent')]"))
            )
            pni_rlagent.click()
            time.sleep(1)
            if self.headless:
                continue_button = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.ID, "submit_approve_access"))
                )
            else:
                continue_button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//*[contains(text(), 'Continue')]"))
                )
            continue_button.click()
            self.driver.switch_to.window(main_window)
        except Exception as e:
            if self.verbose:
                print("An error occurred:", e)
            return False
        return True

    def refresh(self)-> None:
        """Refreshes the current page. Useful to force rendering of neurons."""
        self.driver.refresh()
        if self.verbose:
            print("Page refreshed.")

    def google_login(self, verbose:bool =True)-> None:
        """ 
        Logs into Google account using the mail address and password provided in the constructor. This is highly dependent on the Chrome versions and their updates.
        To tune accordingly.
        Args:
            verbose: Whether to print verbose output.
        Returns:
            None, login is performed in place on the driver instance.
        """
        try:
            self.driver.get('https://accounts.google.com/Login')
            email_input = WebDriverWait(self.driver, 20).until(
                EC.element_to_be_clickable((By.ID, "identifierId"))
            )
            email_input.send_keys(self.mail_address)
            email_input.send_keys(Keys.RETURN)
            password_input = WebDriverWait(self.driver, 20).until(
                EC.element_to_be_clickable((By.XPATH, "//input[@type='password']"))
            )
            password_input.send_keys(self.password)
            password_input.send_keys(Keys.RETURN)
            if verbose:
                print("Login attempted. Waiting for confirmation...")
            WebDriverWait(self.driver, 20).until(EC.url_contains("myaccount.google.com"))
            if verbose:
                print("Login successful!")
        except Exception as e:
            raise Exception(f"Login failed: {e}")
    
    def get_url(self):
        return self.driver.current_url
    
    def change_url(self, user_url: str):
        if self.driver:
            self.driver.get(user_url)  
        else:
            raise Exception("Driver not initialized. Please start the session first.")
    
    def stop_session(self):
        if self.driver:
            self.driver.quit()
            if self.verbose:
                print("Chrome session stopped.")
        else:
            if self.verbose:
                print("No driver instance found.")
    
    def get_screenshot(self, save_path: str = None, resize=False, mouse_x=None, mouse_y=None, fast=True)-> Image.Image:
        """
            Get a screenshot of the current page.
            Args:
                save_path: Path to save the image. If not specified, the image is not saved.
                resize: Boolean to resize the image. Normally we want to resize the image later on before training.
                mouse_x: X coordinate of the mouse. -> Optional, if not specified, the mouse position is not added to the image.
                mouse_y: Y coordinate of the mouse. -> Optional, if not specified, the mouse position is not added to the image.
                fast: Boolean to use the fast method to get the screenshot. If False, the slow method is used (default Selenium method).
        Returns:
            Image.Image, the screenshot of the current page as a PIL Image object.
        """
        if fast:
            screenshot_raw = self.driver.execute_cdp_cmd("Page.captureScreenshot", {"format": "jpeg", "quality": 85})
            screenshot_bytes = base64.b64decode(screenshot_raw["data"])
            image = Image.open(io.BytesIO(screenshot_bytes))
        else:
            screenshot = self.driver.get_screenshot_as_png()
            image = Image.open(io.BytesIO(screenshot))
        if resize:
            image = image.resize((self.resize_width, self.resize_height))
        if mouse_x is not None and mouse_y is not None:
            draw = ImageDraw.Draw(image)
            mouse_y += self.values.neuroglancer_margin_top # Account for the fact we record mouse position from top left of canva and not top left of window
            draw.ellipse((mouse_x - 5, mouse_y - 5, mouse_x + 5, mouse_y + 5), fill='red', outline='red')
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            image.save(save_path, format='PNG')
        return image


    def save_screenshot_bytes(self, save_path: str, quality: int = 85, resize: bool = False)->None:
        """Fastest method to save a screenshot to disk saving the bytes directly.
        Args:
            save_path: The path to save the screenshot to.
            quality: The quality of the image in JPEG compression.
        """
        screenshot_raw = self.driver.execute_cdp_cmd("Page.captureScreenshot", {"format": "jpeg", "quality": quality})
        screenshot_bytes = base64.b64decode(screenshot_raw["data"])
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with Image.open(io.BytesIO(screenshot_bytes)) as image:
            if resize:
                image = image.resize((self.resize_width, self.resize_height), resample=Image.Resampling.LANCZOS)
            image.save(save_path, format="JPEG", quality=quality, optimize=True) 

    def get_JSON_state(self)-> dict:
        """Parse the URL to get the JSON state"""
        script = """
        if (window.viewer && window.viewer.state) {
            return JSON.stringify(viewer.state);
        } else {
            return null;
        }
        """
        try:
            state = self.driver.execute_script(script)
            return state
        except Exception as e:
            if self.verbose:
                print("An error occurred:", e)
            return None

    def change_JSON_state_url(self, json_state, localHost=False)-> None:
        """Change the state of the neuroglancer viewer by changing the URL.
        Args:
            json_state: The JSON state to change to. Takes a dictionary or a string.
            localHost: Whether to use the local host session. Elsewise uses the neuroglancer demo session.
        Returns:
            None, url is changed in place.
        """
        try:
            if isinstance(json_state, dict):
                json_object = json_state
            else:
                json_object = json.loads(json_state)
            serialized_json = json.dumps(json_object)

            encoded_json = urllib.parse.quote(serialized_json)
            if localHost:
                new_url = f"http://localhost:8000/client/#!{encoded_json}"
            else:
                new_url = f"https://neuroglancer-demo.appspot.com/#!{encoded_json}"
            self.change_url(new_url)
        except Exception as e:
            if self.verbose:
                print("An error occurred:", e)

    def change_JSON_state(self, json_state: str)-> None:
        """Changes the state of the neuroglancer viewer through Neuroglancer's API (restoreState)
        Args:
            json_state: The JSON state to change to. Takes a dictionary or a string.
        Returns:
            None, state is changed in place.
        """
        try:
            try:
                json_object = json.loads(json_state)
            except json.JSONDecodeError:
                if self.verbose:
                    print("Invalid JSON state provided.")
                return
            script = f"""
            viewer.state.restoreState({json.dumps(json_object)});
            """
            self.driver.execute_script(script)
        except Exception as e:
            if self.verbose:
                print("An error occurred:", e)
    

    def mouse_key_action(self, x: float, y: float, action: str, keysPressed:str = "None"):
        """Executes a mouse action at a given position. Handling off to MouseActionHandler which executes a JavaScript action.
        Args:
            x: The x coordinate of the mouse.
            y: The y coordinate of the mouse.
            action: The action to execute.
            keysPressed: The keys to press.
        """
        self.action_handler.execute_click(x, y, action, keysPressed)

    def change_viewport_size(self, width: int, height: int):
        self.driver.set_window_size(width, height)
        self.window_width = width
        self.window_height = height

if __name__ == "__main__":
    chrome_ngl = ChromeNGL(headless=False)
    driver = chrome_ngl.driver
    chrome_ngl.start_session()
    chrome_ngl.start_neuroglancer_session(localHost=True)
    time.sleep(1)
    #start_url = "https://neuroglancer-demo.appspot.com/#!%7B%22dimensions%22:%7B%22x%22:%5B4e-9%2C%22m%22%5D%2C%22y%22:%5B4e-9%2C%22m%22%5D%2C%22z%22:%5B4e-8%2C%22m%22%5D%7D%2C%22position%22:%5B138657.265625%2C80856.6953125%2C1335.916015625%5D%2C%22crossSectionScale%22:4.45933655284782%2C%22projectionOrientation%22:%5B0.09884308278560638%2C0.9041123986244202%2C-0.4155852496623993%2C0.009988739155232906%5D%2C%22projectionScale%22:12029.259719517953%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%22precomputed://https://bossdb-open-data.s3.amazonaws.com/flywire/fafbv14%22%2C%22tab%22:%22source%22%2C%22name%22:%22Maryland%20%28USA%29-image%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%22precomputed://gs://flywire_v141_m783%22%2C%22tab%22:%22source%22%2C%22segments%22:%5B%22%21720575940623044103%22%2C%22%21720575940612843473%22%2C%22720575940641265549%22%2C%22720575940625693080%22%2C%22720575940645528430%22%2C%22720575940622572010%22%5D%2C%22name%22:%22flywire_v141_m783%22%7D%5D%2C%22showDefaultAnnotations%22:false%2C%22selectedLayer%22:%7B%22size%22:350%2C%22visible%22:true%2C%22layer%22:%22flywire_v141_m783%22%7D%2C%22layout%22:%22xy-3d%22%7D" 
    start_url = "http://localhost:8000/client/#!%7B%22dimensions%22%3A%20%7B%22x%22%3A%20%5B4e-09%2C%20%22m%22%5D%2C%20%22y%22%3A%20%5B4e-09%2C%20%22m%22%5D%2C%20%22z%22%3A%20%5B4e-08%2C%20%22m%22%5D%7D%2C%20%22layers%22%3A%20%5B%7B%22source%22%3A%20%22precomputed%3A//https%3A//bossdb-open-data.s3.amazonaws.com/flywire/fafbv14%22%2C%20%22type%22%3A%20%22image%22%2C%20%22tab%22%3A%20%22source%22%2C%20%22name%22%3A%20%22Maryland%20%28USA%29-image%22%7D%2C%20%7B%22tab%22%3A%20%22segments%22%2C%20%22source%22%3A%20%22graphene%3A//middleauth%2Bhttps%3A//prodv1.flywire-daf.com/segmentation/1.0/flywire_public%22%2C%20%22type%22%3A%20%22segmentation%22%2C%20%22segments%22%3A%20%5B%22720575940595846828%22%5D%2C%20%22colorSeed%22%3A%20883605311%2C%20%22name%22%3A%20%22Public%20Release%20783-segmentation%22%7D%5D%2C%20%22position%22%3A%20%5B174232.0%2C%2068224.0%2C%203870.0%5D%2C%20%22showDefaultAnnotations%22%3A%20false%2C%20%22perspectiveOrientation%22%3A%20%5B0%2C%200%2C%200%2C%201%5D%2C%20%22projectionScale%22%3A%201500%2C%20%22crossSectionScale%22%3A%200.5%2C%20%22jsonStateServer%22%3A%20%22https%3A//globalv1.flywire-daf.com/nglstate/post%22%2C%20%22selectedLayer%22%3A%20%7B%22layer%22%3A%20%22annotation%22%2C%20%22visible%22%3A%20true%7D%2C%20%22layout%22%3A%20%22xy-3d%22%7D"
    #chrome_ngl.change_url(start_url)
    #chrome_ngl.start_graphene_session(start_url)
    #chrome_ngl.get_screenshot("./screenshot.png", resize=False)
    start_time = time.time()
    urls = []
    input("Press Enter to start collecting URLs...")

    for i in range(300):
        try:
            time.sleep(0.5)
            url = chrome_ngl.get_url()
            print(f"URL {i+1}")
            urls.append(url)
            #chrome_ngl.get_screenshot(f"./proxy/host_images/screenshot_0", resize=True, image_width=960, image_height=540)
            #chrome_ngl.write_screenshot(f"./proxy/host_images/screenshot2.jpeg", resize=True)
            #chrome_ngl.write_screenshot_bytes(f"./proxy/host_images/screenshot_bytes")
            #print(chrome_ngl.get_url())
        except ValueError:
            print("Invalid input. Please enter x and y as integers separated by a comma.")
        except Exception as e:
            print(f"An error occurred: {e}")
    # save urls file
    with open("urls.txt", "w") as f:
        for url in urls:
            f.write(url + "\n")
    print("Time taken: ", time.time() - start_time)
    print("Average time is ", (time.time() - start_time)/100)
