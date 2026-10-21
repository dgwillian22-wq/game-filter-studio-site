# WLSS 3D Video

App paralelo ao WLSS: abre **qualquer vídeo** e mostra em 3D em tempo real, usando a mesma ideia do
"3D sem óculos" (a imagem é deslocada pela profundidade, sem buracos e sem recorte).

- A profundidade vem do modelo **Depth Anything V2 Small** (ONNX) rodando na GPU via **DirectML**
  (qualquer placa no Windows) ou CUDA; sem GPU cai para CPU.
- O desenho é feito em OpenGL: cada pixel procura a superfície mais próxima do olho (ray march),
  então o objeto da frente sempre cobre o fundo.
- A IA roda em outra thread: o vídeo e o efeito não travam, mesmo que a IA seja mais lenta que o vídeo.
  Há mistura no tempo e reinício em corte de cena para a profundidade não piscar.

## Modos (teclas 0 a 5)

| Tecla | Modo | Para quê |
|---|---|---|
| 0 | Original | comparar |
| 1 | **Pisca 3D** | alterna esquerda/direita; o cérebro enxerga profundidade, sem óculos |
| 2 | Vai-e-vem 3D | mesma ideia, movimento suave |
| 3 | Lado a lado | TV / monitor 3D |
| 4 | Anaglifo | óculos vermelho/ciano |
| 5 | Mapa de profundidade | ajustar e depurar |

Outras teclas: `Espaço` pausa, `←` `→` pulam 5 s, `+` `-` força do 3D, `[` `]` convergência
(o que fica na altura da tela), `,` `.` velocidade do pisca, `I` inverte a profundidade,
`F` tela cheia, `O` abre arquivo, `H` mostra/esconde a ajuda, `Esc` sai.
Também dá para arrastar um vídeo para a janela.

## Rodar

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m wlss3d meu_filme.mp4
```

Precisa do `ffmpeg` no PATH (ou `ffmpeg.exe` ao lado do app) para o áudio e para exportar.
Na primeira execução o modelo (~100 MB) é baixado para `%LOCALAPPDATA%\WLSS3D\models`.
Sem internet ou sem o modelo, o app usa uma profundidade aproximada sem IA, só para teste
(`--no-ai` força isso).

Opções úteis: `--mode pisca|vaievem|sbs|anaglifo`, `--strength 0.04`, `--hz 3`,
`--depth-size 308` (menor = mais rápido), `--loop`, `--fullscreen`, `--model arquivo.onnx`.

## Converter um vídeo em arquivo

```bat
python -m wlss3d filme.mp4 --export filme_3d.mp4 --mode sbs
```

## Gerar o .exe

`build_windows.bat` (PyInstaller). Copie o `ffmpeg.exe` para a pasta `dist\WLSS3D`.

## Testes

```bash
pip install pytest
pytest
```

Os testes rodam o renderizador de verdade (OpenGL offscreen), a exportação com áudio, o decodificador,
o relógio de áudio e o suavizador de profundidade. **Não foram testados aqui**: o modelo de IA real
(sem acesso ao servidor do modelo neste ambiente), a placa de som, DirectML e o executável do Windows.
