import cv2
import moviepy.editor as mp
import os

# Definir as variáveis
n_imagens_boas = 10  # Número de imagens boas
n_imagens_ruins = 10  # Número de imagens ruins
fps = 24  # Taxa de quadros por segundo
duracao_video_total = 20  # Duração do vídeo em segundos
duracao_video_indv = 1 # Duração de cada clipe

# Carregar as imagens
imagens_boas = [cv2.imread("MedicineBoxLabelSorter/Generator/Good/"+imagem) for imagem in os.listdir('MedicineBoxLabelSorter/Generator/Good')]
imagens_ruins = [cv2.imread("MedicineBoxLabelSorter/Generator/Bad/"+imagem) for imagem in os.listdir('MedicineBoxLabelSorter/Generator/Bad')]

# Criar clipes de vídeo para imagens boas
clips_boas = [(mp.ImageClip(img).set_duration(0.2)) for img in imagens_boas]

## Com os clipes criados e reservados em memória, vamos criar o slideshow
#Variaveis do clipe
duracao_efeito = 0.1
duracao_clipe = 0.1
# Começamos com a imagem estática inicial e terminamos com a final da mesma forma
first_clip = mp.CompositeVideoClip(
    [
        clips_boas[0]
        .set_pos("center")
        .fx(mp.transfx.slide_out, duration=duracao_efeito, side="left")
    ]
).set_start((duracao_clipe - duracao_efeito) * 0)

# Para o último video começamos da direita para a esquerda
last_clip = (
    mp.CompositeVideoClip(
        [
            clips_boas[-1]
            .set_pos("center")
            .fx(mp.transfx.slide_in, duration=duracao_efeito, side="right")
        ]
    )
    .set_start((duracao_clipe - duracao_efeito) * (len(clips_boas) - 1))
    .fx(mp.transfx.slide_out, duration=duracao_efeito, side="left")
)
videos = (
    [first_clip]
    # Para todos os clipes no meio aplicamos o slide in do último com slide out para o próximo
    + [
        (
            mp.CompositeVideoClip(
                [
                    clip.set_pos("center").fx(
                        mp.transfx.slide_in, duration=duracao_efeito, side="right"
                    )
                ]
            )
            .set_start((duracao_clipe - duracao_efeito) * idx)
            .fx(mp.transfx.slide_out, duration=duracao_efeito, side="left")
        )
        for idx, clip in enumerate(clips_boas[1:-1], start=1)
    ]
    + [last_clip]
)

concat_clip = mp.concatenate_videoclips(videos)
concat_clip.write_videofile("MedicineBoxLabelSorter/Generator/videos/Final.mp4", fps=24)
