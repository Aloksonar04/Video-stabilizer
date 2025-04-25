from flask import Flask, render_template, request, redirect, url_for
import os
from stabilizer import stabilize_video

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'static/output'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET'])
def index():
    return render_template('interface.html', output_video=None)

@app.route('/upload', methods=['POST'])
def upload():
    video = request.files['video']
    if video:
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
        output_filename = f'stabilized_{video.filename}'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        video.save(input_path)
        stabilize_video(input_path, output_path)
        return render_template('interface.html', output_video=output_filename)
    return redirect(url_for('interface'))

if __name__ == '__main__':
    app.run(debug=True)
