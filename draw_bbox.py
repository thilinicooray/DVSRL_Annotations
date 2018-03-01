from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from PIL import Image as PIL_Image

def draw_rectangle(draw, coordinates, color, width=1):
    for i in range(width):
        rect_start = (coordinates[0][0] - i, coordinates[0][1] - i)
        rect_end = (coordinates[1][0] + i, coordinates[1][1] + i)
        draw.rectangle((rect_start, rect_end), outline = color)

        
def draw_bbox_object(image, coor):
    draw = ImageDraw.Draw(image)
    draw_rectangle(draw, ((coor[0], coor[1]),
                    (coor[2], coor[3])), "yellow", 3)
    return image 
    
img = PIL_Image.open('data/1382.png', 'r') 
new_img = draw_bbox_object(img, (238, 254, 238 + 57, 254 + 259))
new_img.show() 

# "h": 259,
#                "merged_object_ids": [],
#                "names": ["man"],
#                "object_id": 3798577,
#                "synsets": ["man.n.01"],
#                "w": 57,
#                "x": 238,
#                "y": 254}