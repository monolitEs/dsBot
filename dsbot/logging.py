
class LogManager:

    def __init__(self, txt_file = None):
        self.txt_file = txt_file

    def logging(self, message):
        if self.txt_file:
            with open(self.txt_file, 'a') as f:
                f.write(message + '\n')
                f.close()
        else:
            print(message)