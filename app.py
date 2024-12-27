from flask import Flask, render_template, jsonify, request
import random
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField  
from werkzeug.utils import secure_filename
import os
from sudokuSolver import *
import cv2 
import numpy as np
from keras.api.models import load_model
import imutils 
from matplotlib import pyplot as plt   
import os    
from PIL import Image   
 
app = Flask(__name__)

app.config["SECRET_KEY"]='supersecretkey'
app.config['UPLOAD_FOLDER']='static/uploadedSudoku'
class uploadFileForm(FlaskForm):
    file = FileField("File")
    submit=SubmitField("Upload Sudoku")


def generate_sudoku(difficulty):
    board = [[0] * 9 for _ in range(9)]
    fill_board(board)

    if difficulty == 'easy':
        remove_count = 30
    elif difficulty == 'medium':
        remove_count = 45
    else:
        remove_count = 60

    for _ in range(remove_count):
        row, col = random.randint(0, 8), random.randint(0, 8)
        while board[row][col] == 0:
            row, col = random.randint(0, 8), random.randint(0, 8)
        board[row][col] = 0

    return board


def fill_board(board):
    num_list = list(range(1, 10))
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                random.shuffle(num_list)
                for num in num_list:
                    if is_safe(board, row, col, num):
                        board[row][col] = num
                        if fill_board(board):
                            return True
                        board[row][col] = 0
                return False
    return True


def is_safe(board, row, col, num):
    for x in range(9):
        if board[row][x] == num or board[x][col] == num:
            return False

    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(3):
        for j in range(3):
            if board[i + start_row][j + start_col] == num:
                return False

    return True


def solve_sudoku(board):
    fill_board(board)
    return board


current_puzzle = generate_sudoku('easy')
current_solution = solve_sudoku([row[:] for row in current_puzzle])


@app.route('/', methods=["GET", "POST" ])
def index():
    form = uploadFileForm()
    if form.validate_on_submit():
        file=form.file.data  #Grab file
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename))) #save file 
        algo() #Run the algorithm

    file_path="static/solvedSudoku/solved.jpg"
    if os.path.exists(file_path):
        print('File exists')
    
    return render_template('index.html', puzzle=current_puzzle, form=form, img=file_path)

@app.route('/static/solvedSudoku/solved.jpg')
def send_uploaded_file(filename=''):
    from flask import send_from_directory
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route('/new_puzzle', methods=['POST'])
def new_puzzle():
        
    data = request.get_json()
    difficulty = data.get('difficulty', 'easy')

    global current_puzzle, current_solution
    current_puzzle = generate_sudoku(difficulty)
    current_solution = solve_sudoku([row[:] for row in current_puzzle])
    return jsonify(current_puzzle)


@app.route('/show_solution', methods=['POST'])
def show_solution():
    return jsonify(current_solution)


@app.route('/check_answer', methods=['POST'])
def check_answer():
    data = request.get_json()
    user_board = data.get('board')

    correct = user_board == current_solution
    return jsonify({'correct': correct})


@app.route('/hint', methods=['POST'])
def hint():
    global current_puzzle, current_solution
    hint_options = [
        (row, col) for row in range(9) for col in range(9)
        if current_puzzle[row][col] == 0
    ]

    if not hint_options:
        return jsonify({})  # No hints available

    row, col = random.choice(hint_options)
    current_puzzle[row][col] = current_solution[row][col]  # Fill in a number from the solution
    return jsonify({'row': row, 'col': col, 'value': current_solution[row][col]})


if __name__ == "__main__":
    app.run(debug=False)