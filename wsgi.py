from server import app

if __name__ == "__main__":
    print('WSGI server running at localhost:4000')
    app.run(host='localhost', port=4000)
