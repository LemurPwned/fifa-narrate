def isProperColor(color):
    if color[0] > 65 or color[0] < 50:
        return False
    if color[1] > 10:
        return False
    if color[2] > 65 or color[2] < 50:
        return False
    return True

def findScores(frame):
    y_max = int(len(frame)/4)
    x_max = int(len(frame[0])/4)
    xpos = [x_max, 0]
    ypos = [y_max, 0]
    for y in range(0, y_max, 2):
        for x in range(0, x_max, 2):
            if isProperColor(frame[y, x]):
                if x < xpos[0]:
                    xpos[0] = x

                if x > xpos[1]:
                    xpos[1] = x

                if y < ypos[0]:
                    ypos[0] = y

                if y > ypos[1]:
                    ypos[1] = y
    checker = [ypos[1] - 4, int(xpos[1] / 2) + int(xpos[0]/2)]
    colors = frame[checker[0], checker[1]]
    teamLName = []
    teamRName = []
    score = []
    if colors[0] > 120 and colors[0] < 140 \
        and colors[1] > 240 \
        and colors[2] < 25 and colors[2] > 5:
            return xpos, ypos
            #for i in range(100):
            #    if isProperColor(

    return None, None

def findSurnames(frame):
    y_minL = int(len(frame)*0.75)
    y_maxL = int(len(frame))
    x_maxL = int(len(frame[0])*0.25)

    xposL = [x_maxL, 0]
    yposL = [y_maxL, y_minL]


    for y in range(y_minL, y_maxL, 2):
        for x in range(0, x_maxL, 2):
            if isProperColor(frame[y, x]):
                if x < xposL[0]:
                    xposL[0] = x

                if x > xposL[1]:
                    xposL[1] = x

                if y < yposL[0]:
                    yposL[0] = y

                if y > yposL[1]:
                    yposL[1] = y

    y_minR = int(len(frame)*0.75)
    y_maxR = int(len(frame))
    x_maxR = int(len(frame[0]))
    x_minR = int(len(frame[0]) * 0.75)

    xposR = [x_maxR, x_minR]
    yposR = [y_maxR, y_minR]

    for y in range(y_minR, y_maxR, 2):
        for x in range(x_minR, x_maxR, 2):
            if isProperColor(frame[y, x]):
                if x < xposR[0]:
                    xposR[0] = x

                if x > xposR[1]:
                    xposR[1] = x

                if y < yposR[0]:
                    yposR[0] = y

                if y > yposR[1]:
                    yposR[1] = y
    
    if xposL[0] == x_maxL and xposL[1] == 0 and yposL[0] == y_maxL and yposL[1] == y_minL:
        xposL = None
        yposL = None

    if xposR[0] == x_maxR and xposR[1] == x_minR and yposR[0] == y_maxR and yposR[1] == y_minR:
        xposR = None
        yposR = None
    return xposL, yposL, xposR, yposR


