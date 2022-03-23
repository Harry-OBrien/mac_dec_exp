import sys

log_verb = 0

INVALID_NODE = sys.maxsize

def set_msg_verb(v):
    global log_verb
    log_verb = v

def get_verb():
    return log_verb
    
def msg(verb, s):
    if (log_verb >= verb):
        for i in range(0,verb):
            print("    ",end='')
        print("LOG: ",end='')
        print(s)
        
def error(s, format):
    print("ERROR: ",end='')
    print(s % format)
    
    sys.exit(1)
    
def error(s):
    print("ERROR: ",end='')
    print(s)
    
    sys.exit(1)
 
    
