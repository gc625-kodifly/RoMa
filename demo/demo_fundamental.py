from PIL import Image
import torch
import cv2
from roma import roma_outdoor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--im_A_path", default="assets/sacre_coeur_A.jpg", type=str)
    parser.add_argument("--im_B_path", default="assets/sacre_coeur_B.jpg", type=str)

    args, _ = parser.parse_known_args()
    im1_path = args.im_A_path
    im2_path = args.im_B_path

    # Create model
    roma_model = roma_outdoor(device=device)


    W_A, H_A = Image.open(im1_path).size
    W_B, H_B = Image.open(im2_path).size

    # Match
    warp, certainty = roma_model.match(im1_path, im2_path, device=device)
    # Sample matches for estimation
    matches, certainty = roma_model.sample(warp, certainty)
    kpts1, kpts2 = roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)    
    F, mask = cv2.findFundamentalMat(
        kpts1.cpu().numpy(), kpts2.cpu().numpy(), ransacReprojThreshold=0.2, method=cv2.USAC_MAGSAC, confidence=0.999999, maxIters=10000
    )
    
    # draw keypoints
    print(type(kpts1))
    
    H, _ = cv2.findHomography(kpts1.cpu().numpy(), kpts2.cpu().numpy(), method=cv2.RANSAC)

    a = 0.7
    # Apply Homography Transformation
    image1 = cv2.imread(im1_path)
    image2 = cv2.imread(im2_path)
    height, width, channels = image2.shape
    warped_image = cv2.warpPerspective(image1, H, (width, height))
    cv2.imshow("transformed", image2)
    cv2.imshow("tgt", warped_image)
    
    cv2.addWeighted(warped_image, a, image2, 1-a, 0, warped_image)
    # Display or save the warped image
    cv2.imshow("Warped Image", warped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    
    