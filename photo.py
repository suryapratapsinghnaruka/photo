import cv2
import numpy as np
from flask import Flask, request, send_file, jsonify, send_from_directory
import io
from PIL import Image
import os
from fpdf  import FPDF
from datetime import datetime
from io import BytesIO
from PyPDF2 import PdfFileWriter, PdfFileReader
app = Flask(__name__)

@app.route('/scan', methods=['POST'])
def scan():
    # Get image file from request
    file = request.files['image']

# Read image file into numpy array
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

# Convert numpy array to PIL Image
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for PIL

# Convert the PIL Image to grayscale
    gray_pil = pil_img.convert('L')

# Convert the grayscale PIL Image to numpy array
    gray = np.array(gray_pil)

# Apply threshold to create a binary image
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour (which is likely to be the document)
    largest_contour = max(contours, key=cv2.contourArea)

# Find the convex hull of the largest contour
    hull = cv2.convexHull(largest_contour)

# Approximate the polygonal curve of the convex hull with a simpler curve
    epsilon = 0.05 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)

# Ensure that the approximated curve has four vertices
    if len(approx) != 4:
        print("Error: The document does not have four corners.")
    else:
    # Find the corners of the approximated curve
        approx = np.squeeze(approx)
    corners = np.zeros((4, 2), dtype=np.float32)
    corners[0] = approx[np.argmin(np.sum(approx, axis=1))]
    corners[2] = approx[np.argmax(np.sum(approx, axis=1))]
    corners[1] = approx[np.argmin(np.diff(approx, axis=1))]
    corners[3] = approx[np.argmax(np.diff(approx, axis=1))]

    # Compute the perspective transform matrix and apply it to the original image
    height = np.sqrt((corners[2][0] - corners[3][0])**2 + (corners[2][1] - corners[3][1])**2)
    width = np.sqrt((corners[1][0] - corners[2][0])**2 + (corners[1][1] - corners[2][1])**2)
    dst_pts = np.array([[0, height-1], [0, 0], [width-1, 0], [width-1, height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(corners, dst_pts)
    warped = cv2.warpPerspective(img, M, (int(width), int(height)))

    warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)





    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

# Estimate the blur kernel (assuming Gaussian blur)
    kernel_size = 5
    blur_kernel = cv2.getGaussianKernel(kernel_size, 0)
    blur_kernel = blur_kernel @ blur_kernel.T

# Perform Wiener deconvolution to recover the sharp image
    psf = np.fft.fft2(blur_kernel, gray.shape)
    gray_fft = np.fft.fft2(gray)
    gray_deconv = np.real(np.fft.ifft2(gray_fft / (psf + 1e-8)))
    gray_deconv = np.uint8(np.clip(gray_deconv, 0, 255))

# Perform unsharp masking to enhance the sharpness of the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    unsharp_mask = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    unsharp_mask = np.uint8(np.clip(unsharp_mask, 0, 255))

    # final_gray = cv2.cvtColor(unsharp_mask, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('Processed Image', unsharp_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Convert warped image to JPEG format
    # ret, jpeg = cv2.imencode('.jpg', unsharp_mask)

    # Encode the processed image (unsharp_mask) to JPEG format
    # ret, jpeg = cv2.imencode('.jpg', unsharp_mask, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    ret, jpeg = cv2.imencode('.jpg', unsharp_mask, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    save_dir = 'static/uploads'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    now = datetime.now()
    save_path = os.path.join(save_dir, f'document_{now.strftime("%Y%m%d_%H%M%S")}.jpg')
    cv2.imwrite(save_path, unsharp_mask)



  # Generate PDF document with saved image
    pdf_path = os.path.join(save_dir, f'document_{now.strftime("%Y%m%d_%H%M%S")}.pdf')
    pdf = FPDF()
    pdf.add_page()
    pdf.image(save_path, 0, 0, pdf.w, pdf.h)
    pdf.output(pdf_path, 'F')

    global pdffileurl
    pdffileurl = pdf_path
     # Return path to saved PDF document in JSON response
    return jsonify({'path': pdf_path})



@app.route('/pdf', methods=['GET'])
def pdf():

    params = request.args.getlist('path')
    print(params)
    # Get path to saved PDF document from request
    path = pdffileurl

    if path is None:
        # Return error response if path is missing
        return {'error': 'Path parameter is missing.'}, 400

        # Get directory of saved PDF document
    directory = os.path.dirname(path)

    # Return PDF document in response
    return send_from_directory(directory, os.path.basename(path), as_attachment=True)
   

if __name__ == '__main__':
    app.run()
