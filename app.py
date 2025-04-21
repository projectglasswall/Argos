import cv2
from flask import Flask, Response, render_template, jsonify, url_for, send_file, request
from ultralytics import YOLO
from djitellopy import Tello # Import Tello
import datetime
import csv
import time
# import pandas as pd # No longer needed for the summary count
import os
import threading # Keep threading for Flask app runner AND flight paths
from collections import Counter # Import Counter for efficient counting

app = Flask(__name__)

# --- Configuration ---
MODEL_PATH = "yolov10n.pt"
CSV_FILENAME = "detection_log.csv"
FRAME_SKIP = 1 # Process every Nth frame
LOG_INTERVAL_SECONDS = 2 # Log detections to CSV interval
JPEG_QUALITY = 85
FLIGHT_SPEED = 20 # Tello speed (increased slightly for path execution)
VERTICAL_ASCENT_CM = 130 # How many cm to move up after takeoff. Adjust as needed.
PATH_MOVEMENT_DISTANCE = 75 # How far to move for path segments (cm), adjust as needed
PATH_SLEEP_INTERVAL = 3 # Seconds to wait between path movements
# ---------------------

# --- Global Variables ---
STREAM_ACTIVE = False # Controls stream and logging
FLIGHT_PATH_ACTIVE = False #Track if a flight path is currently executing
ALLOWED_CLASSES_FILTER = [] # User-selected classes
tello = None # Tello object
detected_objects_buffer = [] # Buffer for CSV logging
last_logged_time = time.time()
max_simultaneous_counts = {} # Dictionary to store max simultaneous counts per class
flight_thread = None # To hold the flight path thread object
# ---------------------

# --- YOLO Model Initialization ---
try:
    print("--- Loading YOLO model... ---")
    model = YOLO(MODEL_PATH)
    if hasattr(model, 'names'): CLASS_NAMES_DICT = model.names
    elif hasattr(model, 'model') and hasattr(model.model, 'names'): CLASS_NAMES_DICT = model.model.names
    else:
        print("⚠️ Warning: Could not auto-determine class names. Using fallback subset.")
        CLASS_NAMES_DICT = {0: 'person', 62: 'tv', 63: 'laptop', 60: 'dining table'} # Example fallback
    AVAILABLE_CLASSES = sorted(list(CLASS_NAMES_DICT.values()))
    print(f"✅ Model loaded. Available classes ({len(AVAILABLE_CLASSES)}): {', '.join(AVAILABLE_CLASSES[:10])}...") # Print only first few
except Exception as e:
    print(f"❌ ERROR: Failed to load YOLO model from {MODEL_PATH}. {e}")
    exit()
# -----------------------------------

# --- Tello Initialization ---
def initialize_tello():
    global tello
    print("--- Initializing Tello... ---")
    tello = Tello()
    try:
        tello.connect()
        print(f"✅ Tello connected. Battery: {tello.get_battery()}%")
        tello.streamon()
        print("✅ Tello stream ON.")
        tello.set_speed(FLIGHT_SPEED)
        print(f"✅ Tello speed set to: {FLIGHT_SPEED} cm/s")
        return True
    except Exception as e:
        print(f"❌ ERROR connecting/starting Tello stream: {e}")
        tello = None
        return False
# --------------------------

# --- CSV Initialization ---
def initialize_csv():
    write_header = not os.path.exists(CSV_FILENAME) or os.path.getsize(CSV_FILENAME) == 0
    if write_header:
        try:
            with open(CSV_FILENAME, "w", newline="") as file:
                writer = csv.writer(file); writer.writerow(["Timestamp", "Class", "Confidence", "X1", "Y1", "X2", "Y2"])
            print(f"✅ Initialized CSV file: {CSV_FILENAME}")
        except IOError as e: print(f"❌ ERROR writing CSV header: {e}"); return False
    return True
# --------------------------

# --- Frame Generation (Reads from Tello, Detects, Updates Max Counts, Logs to CSV) ---
# (No changes needed in generate_frames itself for path logic)
def generate_frames():
    global last_logged_time, detected_objects_buffer, STREAM_ACTIVE, tello, ALLOWED_CLASSES_FILTER, CLASS_NAMES_DICT, max_simultaneous_counts

    if not tello:
        print("--- GEN_FRAMES: Tello not initialized. Cannot generate frames. ---")
        return

    print("--- GEN_FRAMES: Checking for Tello frame reader... ---")
    frame_read = tello.get_frame_read()
    if frame_read is None:
        print("--- GEN_FRAMES: Failed to get Tello frame reader. Cannot generate frames. ---")
        # Attempt to restart stream if Tello object exists
        if tello:
            print("--- GEN_FRAMES: Attempting tello.streamon() again... ---")
            try:
                tello.streamon()
                time.sleep(1) # Give it a moment
                frame_read = tello.get_frame_read()
                if frame_read:
                    print("--- GEN_FRAMES: Successfully re-acquired frame reader. ---")
                else:
                     print("--- GEN_FRAMES: Still failed to get frame reader after retry. Stopping generation. ---")
                     return
            except Exception as stream_err:
                 print(f"--- GEN_FRAMES: Error during streamon retry: {stream_err}. Stopping generation. ---")
                 return
        else:
            return # Exit if no tello object

    print("--- GEN_FRAMES: Starting frame generation loop... ---")
    frame_count = 0
    while True:
        if not STREAM_ACTIVE:
            print("--- GEN_FRAMES: STREAM_ACTIVE is False, breaking loop. ---")
            break

        try:
            frame = frame_read.frame
            if frame is None:
                # print("--- GEN_FRAMES: Got None frame, sleeping... ---") # Can be noisy
                time.sleep(0.05)
                continue

            frame_count += 1
            current_time = time.time()

            # Process Frame Conditionally (Frame Skipping)
            if frame_count % FRAME_SKIP == 0:
                frame_rgb_for_yolo = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model(frame_rgb_for_yolo, stream=False, verbose=False)

                current_frame_detections = [] # Track detections within this frame

                if results and results[0].boxes is not None:
                    boxes = results[0].boxes
                    for box in boxes:
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        cls_name = CLASS_NAMES_DICT.get(cls_id, f"class_{cls_id}")

                        if ALLOWED_CLASSES_FILTER and cls_name not in ALLOWED_CLASSES_FILTER:
                            continue

                        current_frame_detections.append(cls_name)

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        label = f"{cls_name} {conf:.2f}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        detected_objects_buffer.append([
                            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                            cls_name, conf, x1, y1, x2, y2
                        ])

                # Update Max Simultaneous Counts
                if current_frame_detections:
                    frame_counts = Counter(current_frame_detections)
                    for cls_name, count in frame_counts.items():
                        max_simultaneous_counts[cls_name] = max(max_simultaneous_counts.get(cls_name, 0), count)
                    # print(f"--- Frame Counts: {dict(frame_counts)} | Max Counts Updated: {max_simultaneous_counts} ---") # Debug print

                # Log detections periodically TO CSV
                if detected_objects_buffer and current_time - last_logged_time >= LOG_INTERVAL_SECONDS:
                    try:
                        if not os.path.exists(CSV_FILENAME) or os.path.getsize(CSV_FILENAME) == 0: initialize_csv()
                        if os.path.exists(CSV_FILENAME):
                            with open(CSV_FILENAME, "a", newline="") as file: writer = csv.writer(file); writer.writerows(detected_objects_buffer)
                            detected_objects_buffer = []
                            last_logged_time = current_time
                        # else: print(f"--- GEN_FRAMES DEBUG: ERROR - CSV missing during log.") # Can be noisy
                    except IOError as e: print(f"--- GEN_FRAMES DEBUG: ERROR writing CSV log: {e}")

            # Add Battery Overlay
            try:
                # Reduce frequency of battery checks as they can be slow
                if frame_count % 30 == 1: # Check roughly every second (assuming ~30fps)
                     battery = tello.get_battery()
                     last_known_battery = battery # Store it
                cv2.putText(frame, f"Batt: {last_known_battery}%", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            except NameError: # If last_known_battery isn't set yet
                 cv2.putText(frame, "Batt: ?", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            except Exception: # Catch other Tello communication errors
                cv2.putText(frame, "Batt: N/A", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


            # Encode and Yield the frame
            frame_rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Use original BGR frame for display
            ret_encode, buffer = cv2.imencode(".jpg", frame_rgb_display, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ret_encode:
                # print("--- GEN_FRAMES DEBUG: WARNING - Failed to encode frame. ---") # Can be noisy
                continue

            frame_bytes = buffer.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

            time.sleep(0.01) # Small sleep

        except Exception as e:
            print(f"--- GEN_FRAMES: ERROR in frame processing loop: {e} ---")
            # Check if it's a Tello connection issue
            if "Tello disconnected" in str(e) or "command failed" in str(e).lower():
                 print("--- GEN_FRAMES: Tello connection likely lost. Stopping stream. ---")
                 STREAM_ACTIVE = False # Signal stop
                 break # Exit the loop
            time.sleep(0.5)

    print("--- GEN_FRAMES: Frame generation loop finished. ---")
# -------------------------------------------

# --- Flight Path Execution Functions ---
def execute_left_right_path():
    global FLIGHT_PATH_ACTIVE, tello
    print("--- PATH EXEC: Starting Left-to-Right Path ---")
    try:
        if not tello or not STREAM_ACTIVE: return # Safety check

        print(f"--- PATH EXEC: Moving Right {PATH_MOVEMENT_DISTANCE} cm ---")
        tello.move_right(PATH_MOVEMENT_DISTANCE)
        time.sleep(PATH_SLEEP_INTERVAL)
        if not STREAM_ACTIVE: return # Check if stopped during sleep

        print(f"--- PATH EXEC: Moving Left {PATH_MOVEMENT_DISTANCE * 2} cm ---") # Go back past start
        tello.move_left(PATH_MOVEMENT_DISTANCE * 2)
        time.sleep(PATH_SLEEP_INTERVAL)
        if not STREAM_ACTIVE: return

        print(f"--- PATH EXEC: Moving Right {PATH_MOVEMENT_DISTANCE} cm ---") # Return to start column
        tello.move_right(PATH_MOVEMENT_DISTANCE)
        time.sleep(PATH_SLEEP_INTERVAL)

        print("--- PATH EXEC: Left-to-Right Path Finished ---")

    except Exception as e:
        print(f"❌ ERROR during Left-to-Right path execution: {e}")
    finally:
        FLIGHT_PATH_ACTIVE = False # Mark path as finished or aborted

def execute_up_down_path():
    global FLIGHT_PATH_ACTIVE, tello
    print("--- PATH EXEC: Starting Up-and-Down Path ---")
    # Note: Drone is already at VERTICAL_ASCENT_CM height
    try:
        if not tello or not STREAM_ACTIVE: return # Safety check

        print(f"--- PATH EXEC: Moving Up {PATH_MOVEMENT_DISTANCE} cm ---")
        tello.move_up(PATH_MOVEMENT_DISTANCE)
        time.sleep(PATH_SLEEP_INTERVAL)
        if not STREAM_ACTIVE: return

        print(f"--- PATH EXEC: Moving Down {PATH_MOVEMENT_DISTANCE * 2} cm ---") # Go down past start height
        tello.move_down(PATH_MOVEMENT_DISTANCE * 2)
        time.sleep(PATH_SLEEP_INTERVAL)
        if not STREAM_ACTIVE: return

        print(f"--- PATH EXEC: Moving Up {PATH_MOVEMENT_DISTANCE} cm ---") # Return to starting ascended height
        tello.move_up(PATH_MOVEMENT_DISTANCE)
        time.sleep(PATH_SLEEP_INTERVAL)

        print("--- PATH EXEC: Up-and-Down Path Finished ---")
    except Exception as e:
        print(f"❌ ERROR during Up-and-Down path execution: {e}")
    finally:
        FLIGHT_PATH_ACTIVE = False # Mark path as finished or aborted

def run_flight_path(path_name):
    """Selects and runs the appropriate flight path function."""
    global FLIGHT_PATH_ACTIVE
    FLIGHT_PATH_ACTIVE = True # Mark that a path is starting
    print(f"--- Thread started for path: {path_name} ---")
    if path_name == 'left-right':
        execute_left_right_path()
    elif path_name == 'up-down':
        execute_up_down_path()
    else:
        print(f"--- WARNING: Unknown path name '{path_name}' received. No path executed. ---")
        FLIGHT_PATH_ACTIVE = False # Mark inactive if path unknown
    print(f"--- Thread finished for path: {path_name} ---")
# -----------------------------------

# --- Flask Routes ---
@app.route("/")
def index():
    # Pass available class names AND summary data URL
    return render_template("index.html",
                           available_classes=AVAILABLE_CLASSES,
                           summary_data_url=url_for('get_detection_summary'))

@app.route("/get_available_classes") # Endpoint for frontend to fetch classes
def get_available_classes():
     return jsonify({"classes": AVAILABLE_CLASSES})

@app.route("/video_feed")
def video_feed():
    if not tello:
        print("--- VIDEO_FEED: Tello not available, cannot provide feed. ---")
        return Response("Tello not connected or stream error.", status=503)
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/start_stream", methods=['POST'])
def start_stream():
    global STREAM_ACTIVE, tello, ALLOWED_CLASSES_FILTER, detected_objects_buffer, last_logged_time, max_simultaneous_counts, flight_thread, FLIGHT_PATH_ACTIVE
    print("--- Received Start Stream Request (Tello) ---")
    if STREAM_ACTIVE: return jsonify({"status": "already_running", "message": "Stream is already active."}), 400
    if FLIGHT_PATH_ACTIVE: return jsonify({"status": "error", "message": "A flight path is currently active. Please wait or stop."}), 400 # Prevent starting if path running

    if not tello:
         print("--- Tello not initialized, attempting connection... ---")
         if not initialize_tello():
             return jsonify({"status": "error", "message": "Tello connection failed."}), 500
    # Re-check battery before takeoff
    try:
        current_battery = tello.get_battery()
        print(f"--- Battery check before takeoff: {current_battery}% ---")
        if current_battery < 15:
            print(f"❌ ERROR: Battery low ({current_battery}%). Takeoff aborted.")
            return jsonify({"status": "error", "message": f"Battery too low ({current_battery}%) for takeoff."}), 400
    except Exception as e:
        print(f"❌ ERROR checking battery before takeoff: {e}. Aborting start.")
        return jsonify({"status": "error", "message": f"Failed to check battery: {e}"}), 500

    # --- Get Selected Classes and Path ---
    request_data = request.get_json()
    if not request_data: return jsonify({"status": "error", "message": "No data received."}), 400

    selected_classes = request_data.get('selected_classes')
    selected_path = request_data.get('selected_path') # Get the path

    if selected_classes is None or not isinstance(selected_classes, list):
        return jsonify({"status": "error", "message": "Invalid or missing 'selected_classes'."}), 400
    if selected_path is None or selected_path not in ['left-right', 'up-down']: # Validate path
         return jsonify({"status": "error", "message": f"Invalid or missing 'selected_path'. Received: {selected_path}"}), 400

    ALLOWED_CLASSES_FILTER = [cls for cls in selected_classes if cls in AVAILABLE_CLASSES]
    print(f"--- Filtering enabled for classes: {ALLOWED_CLASSES_FILTER} ---")
    print(f"--- Selected path: {selected_path} ---")
    # ---------------------------------------

    if not initialize_csv(): return jsonify({"status": "error", "message": "Failed to initialize log file."}), 500

    # Reset counts and buffers for the new session
    max_simultaneous_counts = {}
    detected_objects_buffer = []
    last_logged_time = time.time()
    print("--- Reset max counts and CSV buffer. ---")

    # --- Start Sequence: Activate Flag -> Takeoff -> Move Up -> START PATH THREAD ---
    STREAM_ACTIVE = True # Activate stream generation

    try:
        print("--- COMMAND: Tello Takeoff ---")
        tello.takeoff()
        print("--- Takeoff command sent. Waiting for stabilization... ---")
        time.sleep(5) # Increase sleep after takeoff

        print(f"--- COMMAND: Tello Move Up by {VERTICAL_ASCENT_CM} cm ---")
        tello.move_up(VERTICAL_ASCENT_CM)
        print("--- Move Up command sent. Waiting briefly... ---")
        time.sleep(3) # Increase sleep after move_up

        # --- Start the selected flight path in a background thread ---
        print(f"--- Starting background thread for path: {selected_path} ---")
        flight_thread = threading.Thread(target=run_flight_path, args=(selected_path,), daemon=True)
        flight_thread.start()
        # -----------------------------------------------------------

    except Exception as e:
        STREAM_ACTIVE = False # Deactivate stream if takeoff/move_up fails
        FLIGHT_PATH_ACTIVE = False # Ensure path marked inactive
        print(f"❌ ERROR during Tello takeoff or move_up: {e}")
        if tello:
            try:
                print("--- ERROR recovery: Attempting land... ---")
                tello.land()
            except Exception as land_e:
                print(f"--- ERROR recovery: Landing attempt also failed: {land_e} ---")
        return jsonify({"status": "error", "message": f"Tello takeoff/move_up failed: {e}"}), 500

    print("✅ Stream marked active. Drone should be hovering higher and starting path.")
    return jsonify({"status": "started"})


@app.route("/stop_stream", methods=['POST'])
def stop_stream():
      global STREAM_ACTIVE, tello, detected_objects_buffer, max_simultaneous_counts, flight_thread, FLIGHT_PATH_ACTIVE
      print("--- Received Stop Stream Request (Tello) ---")
      if not STREAM_ACTIVE and not FLIGHT_PATH_ACTIVE: # Check both flags
          print("--- Stream and Flight Path already stopped. ---")
          return jsonify({"status": "already_stopped"}), 400

      # Signal loops/threads to stop
      STREAM_ACTIVE = False
      FLIGHT_PATH_ACTIVE = False # Signal path thread to stop (though landing takes priority)
      print("--- Stream and Flight Path marked inactive ---")

      # Wait briefly for the flight thread to potentially notice the flag change, but don't block landing
      # if flight_thread and flight_thread.is_alive():
      #      print("--- Waiting briefly for flight thread to potentially stop... ---")
      #      flight_thread.join(timeout=1.0) # Wait max 1 sec

      # Log remaining buffer TO CSV
      if detected_objects_buffer:
          print(f"--- Attempting to log final {len(detected_objects_buffer)} detections to CSV... ---")
          try:
              if not os.path.exists(CSV_FILENAME) or os.path.getsize(CSV_FILENAME) == 0: initialize_csv()
              if os.path.exists(CSV_FILENAME):
                  with open(CSV_FILENAME, "a", newline="") as file: writer = csv.writer(file); writer.writerows(detected_objects_buffer)
                  detected_objects_buffer = []
                  print("--- Final CSV logs written. ---")
              # else: print(f"--- ERROR: CSV missing for final log. ---") # Can be noisy
          except IOError as e: print(f"--- ERROR writing final CSV log: {e} ---")
      else:
          print("--- CSV Buffer empty, no final CSV logs. ---")

      # Land the Drone (This should interrupt any ongoing movement)
      land_success = False
      if tello:
          try:
              print("--- COMMAND: Tello Land ---")
              tello.land()
              print("--- Tello land command sent ---")
              # Don't wait here, let the response go back
              land_success = True
          except Exception as e:
              print(f"❌ ERROR sending Tello land command: {e}")
      else:
            print("--- Tello object not found, cannot send land command. ---")

      print("✅ Stop stream processing complete.")
      # Return final counts
      return jsonify({"status": "stopped", "final_max_counts": max_simultaneous_counts})

@app.route('/get_detection_summary')
def get_detection_summary():
    """Returns the maximum simultaneous detection counts recorded."""
    global max_simultaneous_counts
    print(f"--- Received Get Detection Summary Request. Data: {max_simultaneous_counts} ---")
    return jsonify(max_simultaneous_counts)

# Optional: Keep endpoint for direct CSV download
@app.route("/get_csv")
def get_csv():
    try: return send_file(CSV_FILENAME, mimetype='text/csv', as_attachment=True, download_name='detection_log.csv')
    except FileNotFoundError: return "CSV file not found.", 404
    except Exception as e: print(f"❌ ERROR serving CSV file: {e}"); return "Error serving CSV file.", 500

# --- Main Execution ---
if __name__ == "__main__":
    if not initialize_tello():
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! Tello initialization failed. Check connection/power.   !!!")
        print("!!! Server running but drone features WILL BE unavailable. !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    print("\n-----------------------------------------")
    print(" Starting Tello Object Detection Server ")
    print(" (Mode: Select Path/Classes -> Takeoff/Ascend -> Execute Path -> Land -> Compare Max Counts) ") # Updated mode
    print(f" (Vertical Ascent: {VERTICAL_ASCENT_CM} cm | Path Distance: {PATH_MOVEMENT_DISTANCE} cm)")
    print("-----------------------------------------")
    try:
        # Use development server with threaded=True for handling concurrent requests (video feed + commands)
        # Set debug=False for production or when testing drone stability
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n--- Received KeyboardInterrupt (Ctrl+C) ---")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred running the server: {e}")
    finally:
        # --- Graceful Tello Shutdown ---
        print("\n--- Initiating Tello shutdown sequence ---")
        STREAM_ACTIVE = False # Signal threads/loops
        FLIGHT_PATH_ACTIVE = False

        if tello:
            try:
                 is_flying = False
                 # Attempt to check connection state before landing
                 try:
                     batt = tello.get_battery()
                     print(f"--- Final Battery Check: {batt}% ---")
                     # A successful battery check implies connection and likely flight state
                     is_flying = True
                 except Exception as conn_err:
                     print(f"--- Could not confirm Tello connection state ({conn_err}). Attempting land anyway... ---")
                     # We should still try to land if the object exists, even if connection check fails
                     is_flying = True # Assume we should try landing

                 if is_flying:
                     print("--- Attempting final land command... ---")
                     try:
                         tello.land()
                         time.sleep(3) # Give it time to land
                         print("--- Final land command sent. ---")
                     except Exception as e: print(f"--- Error during final land: {e}")

                 # Turn stream off regardless of flight state if connected
                 try:
                     print("--- Turning stream OFF ---")
                     tello.streamoff()
                 except Exception as e: print(f"--- Error turning stream off: {e}")

                 # End connection
                 try:
                     print("--- Ending Tello connection ---")
                     tello.end()
                 except Exception as e: print(f"--- Error ending connection: {e}")

                 print("--- Tello object cleanup attempted. ---")

            except Exception as outer_e:
                 print(f"--- Error during outer Tello cleanup block: {outer_e} ---")
        else:
             print("--- Tello object was not initialized, no cleanup needed. ---")

        print("--- Tello shutdown sequence complete ---")
        print(f"--- Final Max Simultaneous Counts Recorded: {max_simultaneous_counts} ---")
        print("✅ Flask server stopped.")
        print("-----------------------------------------")