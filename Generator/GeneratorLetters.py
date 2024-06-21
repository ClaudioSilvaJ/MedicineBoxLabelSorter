from PIL import Image, ImageDraw, ImageFont
import os

font_path = r'Generator\5by7.ttf'

font_size = 72
font_color = (0, 0, 0)
alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


if not os.path.exists('Generator/imagens_letras'):
    os.makedirs('Generator/imagens_letras')

font = ImageFont.truetype(font_path, font_size)

def generate_letters():
    for letter in alphabet:
        image = Image.new('RGB', (font_size, font_size), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        draw.text((0, 0), letter, font=font, fill=font_color)
        image.save(f'Generator/imagens_letras/{letter}.png')

