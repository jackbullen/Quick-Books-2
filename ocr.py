import os
import json

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

from dotenv import load_dotenv
load_dotenv()

# Make Computer Vision Client
cv_client = ImageAnalysisClient(
    endpoint=os.getenv("endpoint"),
    credential=AzureKeyCredential(key=os.getenv("key")))

# Send the image to the client and get READ VisualFeatures
with open("old/test2.jpg", "rb") as f:
    img_bytes = f.read()
    result = cv_client.analyze(image_data=img_bytes,
                               visual_features=[VisualFeatures.READ,
                                                VisualFeatures.OBJECTS])

# Save result to JSON
with open("result.json", "w") as f:
    f.write(json.dumps(result.as_dict(), indent=4))


# Annotate results onto the image
# image = Image.open("old/test2.jpg")
# fig = plt.figure(figsize=(image.width/100, image.height/100))
# plt.axis("off")
# draw = ImageDraw.Draw(image)
# font = ImageFont.truetype("/Library/Fonts/Arial Unicode.ttf", size=24)

# if result.read is not None:
#     for line in result.read.blocks[0].lines:
#         r = line.bounding_polygon
#         bounds = ((r[0].x, r[0].y), (r[1].x, r[1].y), (r[2].x, r[2].y), (r[3].x, r[3].y))
#         draw.polygon(bounds, outline='cyan', width=3)
#         draw.text((r[0].x, r[0].y+10), 
#                   line.text, 
#                   fill='red', 
#                   direction="ttb", 
#                   font=font)

# if result.objects is not None:
#     for detected_obj in result.objects.list:
#         r = detected_obj.bounding_box
#         bbox = ((r.x, r.y, (r.x+r.width, r.y+r.height)))
#         draw.rectangle(bbox, outline='cyan', width=3)
    
# plt.imshow(image)
# fig.savefig("analyzed_easy.jpg")