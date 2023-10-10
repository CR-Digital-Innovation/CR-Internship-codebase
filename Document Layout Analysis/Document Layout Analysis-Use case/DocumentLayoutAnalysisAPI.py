from flask import Flask, request, jsonify
import numpy as np
import pdf2image
import cv2
import layoutparser as lp
import os
from flask_restful import Resource, Api

# Initialize layout_analyzer_model and other configurations
layout_analyzer_model = lp.Detectron2LayoutModel('lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config',
                                 extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                                 label_map={1: "TextRegion", 2: "ImageRegion", 3: "TableRegion", 4: "MathsRegion", 5: "SeparatorRegion", 6: "OtherRegion"})

app = Flask(__name__)
api = Api(app)
app.config['UPLOAD_FOLDER'] = 'static'

class LayoutAnalyzer(Resource):
    def post(self):
        try:
            file = request.files['file']
            if file:
                if file.filename.lower().endswith('.pdf'):
                    return self.analyze_layout_pdf(file)
                elif file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    return self.analyze_layout_image(file)
                else:
                    return jsonify({'error': 'Invalid file format. Please upload a PDF or an image.'})
        except Exception as e:
            return jsonify({'error': str(e)})

    def analyze_layout_pdf(self, pdf_file):
        try:
            pdf_file_path = 'temp.pdf'
            pdf_file.save(pdf_file_path)
            images = pdf2image.convert_from_path(pdf_file_path)
            os.remove(pdf_file_path)

            layout_results = []
            for page_number, img in enumerate(images, start=1):
                img_array = np.asarray(img)
                layout_result = layout_analyzer_model.detect(img_array)

                layout_info = []
                for block in layout_result._blocks:
                    block_data = {
                        'x_1': block.block.x_1,
                        'y_1': block.block.y_1,
                        'x_2': block.block.x_2,
                        'y_2': block.block.y_2,
                        'type': block.type,
                        'score': block.score,
                    }
                    layout_info.append(block_data)

                layout_results.append({'page_number': page_number, 'layout_result': layout_info})

            response = {}
            for entry in layout_results:
                page_number = entry['page_number']
                page_layout_result = entry['layout_result']
                response[f'layout_result_page_{page_number}'] = page_layout_result

            return jsonify(response)
        except Exception as e:
            return jsonify({'error': str(e)})

    def analyze_layout_image(self, image):
        try:
            img = cv2.imread(image.filename)
            layout_result = layout_analyzer_model.detect(img)

            layout_info = self.extract_layout_info(layout_result)
            return jsonify({'layout': layout_info})
        except Exception as e:
            return jsonify({'error': str(e)})

    def extract_layout_info(self, layout_result):
        layout_info = []
        for block in layout_result._blocks:
            block_data = {
                'x_1': block.block.x_1,
                'y_1': block.block.y_1,
                'x_2': block.block.x_2,
                'y_2': block.block.y_2,
                'type': block.type,
                'score': block.score,
            }
            layout_info.append(block_data)
        return layout_info


api.add_resource(LayoutAnalyzer, '/api/layout-analyzer')

if __name__ == '__main__':
    app.run(debug=True)
