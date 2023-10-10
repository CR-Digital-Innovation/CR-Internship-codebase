# ... (your existing imports)
from flask import Flask, render_template, request, jsonify, redirect 
import json
from flask import make_response 
import numpy as np
import pdf2image
import cv2
import layoutparser as lp # Assuming lp contains the function to draw boxes on the image
import os
from PIL import Image
import base64
import datetime
import io
from werkzeug.utils import secure_filename


layout_analyzer_model = lp.Detectron2LayoutModel('lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config',
                                 extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                                 label_map={1:"TextRegion", 2:"ImageRegion", 3:"TableRegion", 4:"MathsRegion", 5:"SeparatorRegion", 6:"OtherRegion"})


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'
app.config['DATA'] = []


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            if file.filename.lower().endswith('.pdf'):
                results = analyze_layout_pdf(file)
            elif file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                results = analyze_layout_image(file)
            else:
                return jsonify({'error': 'Invalid file format. Please upload a PDF or an image.'})
            
            if 'error' in results[0]:
                return jsonify(results[0])
            
            app.config['DATA'].extend(results)  # Extend the list with the new results
            
            return render_template('index1.html', entries=app.config['DATA'])
    
    return render_template('index1.html', entries=app.config['DATA'])


def analyze_layout_pdf(pdf_file):
    try:
        # Convert the PDF to a list of images (one image per page)
        pdf_file_path = 'temp.pdf'
        pdf_file.save(pdf_file_path)

        pdf_images = pdf2image.convert_from_path(pdf_file_path)

        # After processing, remove the temporary PDF file
        os.remove(pdf_file_path)

        all_layout_results = []

        # Analyze the entire PDF and calculate the number of pages
        pdf_layout_result = []

        for page_number, pdf_image in enumerate(pdf_images, start=1):
            img = np.asarray(pdf_image)
            layout_result = layout_analyzer_model.detect(img)
            img_with_boxes = lp.draw_box(img, layout_result, box_width=5, box_alpha=0.2, show_element_type=True)

            img_buffer = io.BytesIO()
            img_with_boxes.save(img_buffer, format='PNG')
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

            page_layout_data = []
            for block in layout_result._blocks:
                block_data = {
                    'x_1': block.block.x_1,
                    'y_1': block.block.y_1,
                    'x_2': block.block.x_2,
                    'y_2': block.block.y_2,
                    'type': block.type,
                    'score': block.score,
                }
                page_layout_data.append(block_data)

            pdf_layout_result.append({'page_number': page_number, 'preview': img_base64, 'layout_result': page_layout_data})

        # Calculate the number of pages
        num_pages = len(pdf_images)

        # Add entry for the entire PDF with the number of pages
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        entry = {
            'filename': pdf_file.filename,
            'num_pages': num_pages,
            'preview': None,
            'timestamp': timestamp,
            'layout_result': pdf_layout_result
        }
        all_layout_results.append(entry)

        return all_layout_results

    except Exception as e:
        return [{'error': str(e)}]

def analyze_layout_image(image):
    try:

        filename = secure_filename(image.filename)
        uploaded_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(uploaded_image_path)

        # Read the image using the absolute path
        img = cv2.imread(uploaded_image_path)


        # Read the image as bytes and convert it to an image array
        # img = cv2.imread(image.filename)

        layout_result = layout_analyzer_model.detect(img)
        all_layout_results = []
        pdf_layout_result = []
        blocks_data = []

        for block in layout_result._blocks:
            block_data = {
                'x_1': block.block.x_1,
                'y_1': block.block.y_1,
                'x_2': block.block.x_2,
                'y_2': block.block.y_2,
                'type': block.type,
                'score': block.score,
            }
            blocks_data.append(block_data)

            

        # Draw bounding boxes on the image
        img_with_boxes = lp.draw_box(img, layout_result, box_width=5, box_alpha=0.2, show_element_type=True)

        img_buffer = io.BytesIO()
        img_with_boxes.save(img_buffer, format='PNG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        pdf_layout_result.append({'page_number': 1, 'preview': img_base64, 'layout_result': blocks_data})
        # Add entry to app.config['DATA']
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        entry = {
            'filename': image.filename,
            'num_pages': 1,
            'preview': img_base64,
            'timestamp': timestamp,
            'layout_result': pdf_layout_result
        }
        all_layout_results.append(entry)

        return all_layout_results
    except Exception as e:
        return {'error': str(e)}

@app.route('/download_layout/<filename>', methods=['GET'])
def download_layout(filename):
    try:
        # Find the entry in app.config['DATA'] with the matching filename
        entry = next((e for e in app.config['DATA'] if e['filename'] == filename), None)

        if entry:
            layout_data = {
                'filename': filename,
                'timestamp': entry.get('timestamp'),
  # Use the dictionary structure from the analyze function
            }
            for page_info in entry['layout_result']:
                page_number = page_info['page_number']
                page_layout_result = page_info['layout_result']
                layout_data[f'layout_result_page_{page_number}'] = page_layout_result

            layout_json = json.dumps(layout_data, indent=4)
            response = make_response(layout_json)
            response.headers['Content-Disposition'] = f'attachment; filename={filename.split(".")[0]}.json'
            response.headers['Content-Type'] = 'application/json'
            return response

        return jsonify({'error': 'Layout result data not found for this file.'})

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/view_all_images/<filename>', methods=['GET'])
def view_all_images(filename):
    try:
        entry = next((e for e in app.config['DATA'] if e['filename'] == filename), None)

        if entry:
            image_gallery_html = '<div class="image-gallery">'

            for page_info in entry['layout_result']:
                page_number = page_info['page_number']
                preview_data = page_info.get('preview')

                if preview_data:
                    image_html = f'<img src="data:image/png;base64,{preview_data}" alt="Page {page_number}">'
                    image_gallery_html += image_html

            image_gallery_html += '</div>'

            return render_template('image_gallery.html', image_gallery_html=image_gallery_html, entry=entry)

        return jsonify({'error': 'Layout result data not found for this file.'})

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/clear', methods=['POST'])
def clear_data():
    app.config['DATA'] = []
    return redirect('/')

# ... (your existing main block)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80,debug=True)
