import cv2
import numpy as np

def get_gaze_direction(eye_region):
    gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    _, thresh = cv2.threshold(
        gray, 70, 255, cv2.THRESH_BINARY_INV
    )

    h, w = thresh.shape

    '''
    thresh
        altura h
            +--------------------------+
            |                          |
            |                          |
            |                          |
            +--------------------------+
                                    largura w

        Cada pixel é preto ou branco (0 ou 255).
    '''

    # mirror image: left <-> right 

    right = thresh[:, 0:w//3] 
    #[todas as linhas, 1a coluna(0) até 1/3 da width] zona esquerda da img (direita pq img espelha)]
    
    center = thresh[:, w//3:2*w//3]
    #[todas as linhas, ini: 1/3 width end: 2/3 de width -> centro do olho]

    left = thresh[:, 2*w//3:w]
    #[todas as linhas, ini: 2/3 width ate ao fim da width -> parte direita do olho (esquerda pq img espelha)]

    #contagem dos pixeis brancos dentro destas zonas
    left_count = cv2.countNonZero(left)
    center_count = cv2.countNonZero(center)
    right_count = cv2.countNonZero(right)

    if left_count > center_count and left_count > right_count:
        return "LEFT"
    elif right_count > left_count and right_count > center_count:
        return "RIGHT"
    else:
        return "CENTER"
