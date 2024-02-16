from core import app, db
from waitress import serve

with app.app_context():
    db.create_all()

mode = 'dev'

if __name__ == '__main__':
    if (mode == 'dev'):
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        serve(app, host='0.0.0.0', port=8000, threads=4)
