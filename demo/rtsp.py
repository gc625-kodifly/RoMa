import cv2
import torch
from PIL import Image
from roma import roma_outdoor
import scipy.linalg
import numpy as np
def read_frame_from_stream(cap):
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        return None
    return frame

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





if __name__ == "__main__":
    # RTSP URLs
    rtsp_url1 = "rtsp://admin:$Kodifly2022@192.168.1.139"
    rtsp_url2 = "rtsp://admin:$Kodifly2022@192.168.1.138"

    # Create capture objects for RTSP streams
    cap1 = cv2.VideoCapture(rtsp_url1)
    cap2 = cv2.VideoCapture(rtsp_url2)

    # Check if the stream captures are opened correctly
    if not cap1.isOpened() or not cap2.isOpened():
        print("Could not open one or both streams")
        exit(1)

    # Create ROMA model
    roma_model = roma_outdoor(device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

    avg_h = None
    counter = 0
    
    try:
        while True:
            # Read frames from the RTSP streams
            frame1 = read_frame_from_stream(cap1)
            frame2 = read_frame_from_stream(cap2)

            # Ensure frames were captured
            if frame1 is None or frame2 is None:
                print("Failed to capture frames from streams")
                break

            # Get the dimensions of the frames
            H_A, W_A, _ = frame1.shape
            H_B, W_B, _ = frame2.shape

            # Convert frames to PIL images for ROMA processing
            im1_pil = Image.fromarray(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
            im2_pil = Image.fromarray(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))

            # Match
            warp, certainty = roma_model.match(im1_pil, im2_pil, device=device)
            # Sample matches for estimation
            matches, certainty = roma_model.sample(warp, certainty)
            
            mask = certainty > 0.9
            matches = matches[mask]
            certainty = certainty[mask]

            sorted_matches = matches[torch.argsort(certainty,descending=True)]
            kpts1, kpts2 = roma_model.to_pixel_coordinates(sorted_matches, H_A, W_A, H_B, W_B)

            # Calculate Homography
            H, _ = cv2.findHomography(kpts1.cpu().numpy(), kpts2.cpu().numpy(), method=cv2.RANSAC)
            print(H)
            lie_algebra = scipy.linalg.logm(H)
            print('='*20)
            print('lie',lie_algebra)
            print('='*20)
            # Apply Homography Transformation
            warped_image = cv2.warpPerspective(frame1, H, (W_B, H_B))

            # Blend Images
            alpha = 0.7
            blended_image = cv2.addWeighted(warped_image, alpha, frame2, 1 - alpha, 0)

            # Display Blended Image
            cv2.imshow("Blended Image", blended_image)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Release resources
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()
