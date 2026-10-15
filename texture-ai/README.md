# TexAI — texturas de jogos antigos remasterizadas com IA

Pega texturas pequenas e borradas (64×64, 128×128, JPEG estourado de jogos de PS1/PS2/PC antigo)
e devolve versões 2×, 4× ou 8× maiores, nítidas e limpas, usando modelos de super-resolução
(Real-ESRGAN por padrão; aceita qualquer modelo ESRGAN/SwinIR/HAT do [OpenModelDB](https://openmodeldb.info)).

Roda **100% local** (sem enviar nada para a internet, exceto o download único dos modelos).

## O que ele faz de específico para texturas

| Problema em texturas | O que o TexAI faz |
|---|---|
| Halo escuro nas bordas de texturas recortadas (grades, folhas) | Espalha a cor das bordas para a área transparente antes do upscale e escala o alfa separado |
| Emenda visível quando a textura se repete (chão, parede) | `--tileable`: processa com a imagem "enrolada" e corta depois |
| Mapas de normais / spec / máscara estragados pela IA | Detecta pelo nome (`_n`, `_normal`, `_spec`, `_mask`...) e usa upscale clássico; normais são renormalizados |
| Pastas gigantes de jogo | Processa recursivamente, mantém a estrutura de subpastas, pula o que já foi feito, um arquivo ruim não derruba o lote |
| Formatos antigos | Lê PNG, JPG, BMP, TGA, DDS, PCX, TIFF, WEBP; salva PNG, TGA, BMP, JPG, DDS, WEBP |
| IA "inventando demais" | `--strength 0.6` mistura a IA com o upscale simples |

## Instalação

```bash
cd texture-ai
pip install -r requirements.txt      # Pillow, numpy, torch, spandrel
```

Sem placa de vídeo funciona na CPU (mais lento). Com NVIDIA/Apple Silicon usa GPU sozinho.
Só quer testar sem PyTorch? `pip install pillow numpy` e use `--model classic`.

## Uso

### Interface gráfica (antes/depois)

```bash
python -m texai serve
```

Abre `http://127.0.0.1:8765`: arraste a textura, escolha modelo/escala, compare com o slider e baixe o PNG.

### Linha de comando

```bash
# uma textura
python -m texai upscale parede.png                  # -> parede_ai.png (4x)

# pasta inteira do jogo, mantendo subpastas -> pasta_ai/
python -m texai upscale C:/jogo/texturas -s 4

# chão/parede que se repete, sem emenda, só 2x, com um toque de nitidez
python -m texai upscale chao.png -s 2 --tileable --sharpen 0.4

# cel shading / desenho
python -m texai upscale personagens/ -m anime

# saída em TGA, IA mais discreta
python -m texai upscale pasta/ --format tga --strength 0.7

# seu próprio modelo do OpenModelDB
python -m texai upscale pasta/ -m C:/modelos/4x_NMKD-Superscale.pth

python -m texai models --download                    # baixa os modelos de uma vez
```

Modelos incluídos (baixados na primeira vez em `~/.cache/texai/models`, com verificação SHA-256):

| `-m` | Para quê |
|---|---|
| `general` (padrão) | Leve e rápido; bom para texturas pequenas com artefatos de compressão |
| `photo` | Máxima qualidade para texturas realistas; pesado |
| `anime` | Cel shading, desenho, cores chapadas |
| `classic` | Sem IA (Lanczos + nitidez); instantâneo, sem PyTorch |

## Dicas

- Memória estourando? `--tile 128` (ou 256) processa em blocos.
- Não carregue arquivos `.pth` de fontes que você não conhece: modelos são código serializado.
- Texturas de jogos geralmente têm **mipmaps** e vão de volta ao jogo por **mod loaders** (Dolphin, PCSX2/DuckStation,
  Skyrim/Fallout, Morrowind etc.). Mantenha os mesmos nomes de arquivo (use `--format same`) para substituir 1:1.
- Respeite a licença do jogo: use para uso pessoal ou mods permitidos; não redistribua assets originais.

## Testes

```bash
pip install pytest && python -m pytest
```
