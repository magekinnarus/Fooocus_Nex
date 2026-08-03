# Nexfocus Prompt Presets

Nexfocus ships 16 prompt presets that rewrite what is actually sent to the
model. Each preset is a template: selecting it adds ready-made prompt lines,
negative guidance, or both around your own prompt. This reference lists the
exact expansion text so you can see what a preset does before you select it.

How presets work:

- In the UI they live in the **Models** panel under **Prompt Presets**. The
  search box filters the list, and several presets can be active at once. The
  default selections are **Fooocus Enhance** and **Fooocus Sharp**.
- Positive text containing `{prompt}` inserts your prompt at that placeholder.
  Positive text without `{prompt}` is added as independent guidance alongside
  your prompt. A preset with an empty positive field adds no positive text.
- Every selected preset also contributes its negative text to the negative
  prompt, on top of your own negative prompt.
- **Random Style** is not a preset. It picks one of the presets at random at
  generation time.
- The shipped presets live in `sdxl_styles/sdxl_styles_fooocus.json` and
  `sdxl_styles/sdxl_styles_dj.json`. Custom JSON files placed in the
  `sdxl_styles/` folder are loaded as additional presets and follow the same
  format.

See [USAGE.md](USAGE.md) for the practical guidance on prompt ownership and
preset selection.

## Presets with Positive Guidance

### Fooocus Sharp

- **Positive:** `cinematic still {prompt} . emotional, harmonious, vignette,
  4k epic detailed, shot on kodak, 35mm photo, sharp focus, high budget,
  cinemascope, moody, epic, gorgeous, film grain, grainy`
- **Negative:** `anime, cartoon, graphic, (blur, blurry, bokeh), text,
  painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly,
  disfigured`

### Fooocus Masterpiece

- **Positive:** `(masterpiece), (best quality), (ultra-detailed), {prompt},
  illustration, disheveled hair, detailed eyes, perfect composition, moist
  skin, intricate details, earrings`
- **Negative:** `longbody, lowres, bad anatomy, bad hands, missing fingers,
  pubic hair,extra digit, fewer digits, cropped, worst quality, low quality`

### Fooocus Photograph

- **Positive:** `photograph {prompt}, 50mm . cinematic 4k epic detailed 4k
  epic detailed photograph shot on kodak detailed cinematic hbo dark moody,
  35mm photo, grainy, vignette, vintage, Kodachrome, Lomography, stained,
  highly detailed, found footage`
- **Negative:** `Brad Pitt, bokeh, depth of field, blurry, cropped, regular
  face, saturated, contrast, deformed iris, deformed pupils, semi-realistic,
  cgi, 3d, render, sketch, cartoon, drawing, anime, text, cropped, out of
  frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid,
  mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn
  face, mutation, deformed, dehydrated, bad anatomy, bad proportions, extra
  limbs, cloned face, disfigured, gross proportions, malformed limbs, missing
  arms, missing legs, extra arms, extra legs, fused fingers, too many
  fingers, long neck`

### Fooocus Cinematic

- **Positive:** `cinematic still {prompt} . emotional, harmonious, vignette,
  highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous,
  film grain, grainy`
- **Negative:** `anime, cartoon, graphic, text, painting, crayon, graphite,
  abstract, glitch, deformed, mutated, ugly, disfigured`

### Fooocus Pony

- **Positive:** `score_9, score_8_up, score_7_up, {prompt}`
- **Negative:** `score_6, score_5, score_4`

### DJ hyperrealism

- **Positive:** `hyperrealistic art, extremely high-resolution details,
  photographic, realism pushed to extreme, fine texture, incredibly lifelike`
- **Negative:** `anime, manga, drawings, abstract, unrealistic, low
  resolution`

### DJ Illustrious

- **Positive:** `masterpiece, best quality, amazing quality, very aesthetic,
  absurdres, newest`
- **Negative:** `bad quality, worst quality, worst detail, sketch, censored,
  watermark, signature`

### DJ Dynamic

- **Positive:** `dynamic pose, interesting angle, eye catching composition,
  depth of field, forced perspective`
- **Negative:** none

### DJ Digital Illustration

- **Positive:** `Digital illustration, perfect composition, intricate
  details`
- **Negative:** `lowres, bad anatomy`

### DJ Pony

- **Positive:** `(score_9), score_8_up, score_7_up, rating_explicit`
- **Negative:** `source_furry, source_pony, score_6, score_5, score_4, low
  quality, bad quality, muscular, furry`

### DJ dystopian

- **Positive:** `dystopian style, bleak, post-apocalyptic, somber, dramatic,
  highly detailed`
- **Negative:** `ugly, deformed, noisy, blurry, low contrast`

### DJ fairy tale

- **Positive:** `fairy tale, magical, fantastical, enchanting, storybook
  style, highly detailed`
- **Negative:** `modern, ordinary, mundane`

## Negative-Only Presets

These presets add no positive text. Their entire effect is the negative
guidance they contribute.

### Fooocus Enhance

- **Negative:** `(worst quality, low quality, normal quality, lowres, low
  details, oversaturated, undersaturated, overexposed, underexposed,
  grayscale, bw, bad photo, bad photography, bad art:1.4), (watermark,
  signature, text font, username, error, logo, words, letters, digits,
  autograph, trademark, name:1.2), (blur, blurry, grainy), morbid, ugly,
  asymmetrical, mutated malformed, mutilated, poorly lit, bad shadow, draft,
  cropped, out of frame, cut off, censored, jpeg artifacts, out of focus,
  glitch, duplicate, (airbrushed, cartoon, anime, semi-realistic, cgi,
  render, blender, digital art, manga, amateur:1.3), (3D ,3D Game, 3D Game
  Scene, 3D Character:1.1), (bad hands, bad anatomy, bad body, bad face, bad
  teeth, bad arms, bad legs, deformities:1.3)`

### Fooocus Semi Realistic

- **Negative:** `(worst quality, low quality, normal quality, lowres, low
  details, oversaturated, undersaturated, overexposed, underexposed, bad
  photo, bad photography, bad art:1.4), (watermark, signature, text font,
  username, error, logo, words, letters, digits, autograph, trademark,
  name:1.2), (blur, blurry, grainy), morbid, ugly, asymmetrical, mutated
  malformed, mutilated, poorly lit, bad shadow, draft, cropped, out of frame,
  cut off, censored, jpeg artifacts, out of focus, glitch, duplicate, (bad
  hands, bad anatomy, bad body, bad face, bad teeth, bad arms, bad legs,
  deformities:1.3)`

### Fooocus Negative

- **Negative:** `deformed, bad anatomy, disfigured, poorly drawn face,
  mutated, extra limb, ugly, poorly drawn hands, missing limb, floating
  limbs, disconnected limbs, disconnected head, malformed hands, long neck,
  mutated hands and fingers, bad hands, missing fingers, cropped, worst
  quality, low quality, mutation, poorly drawn, huge calf, bad hands, fused
  hand, missing hand, disappearing arms, disappearing thigh, disappearing
  calf, disappearing legs, missing fingers, fused fingers, abnormal eye
  proportion, Abnormal hands, abnormal legs, abnormal feet, abnormal fingers,
  drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry,
  soft, deformed, ugly, anime, cartoon, graphic, text, painting, crayon,
  graphite, abstract, glitch`

### DJ Negative Enhance

- **Negative:** `worst quality, low quality, lowres, bw, bad art, watermark,
  signature, text, username, logo, trademark, blur, blurry, grainy, ugly,
  mutated, mutilated, bad shadow, draft, cropped, out of frame, cut off,
  censored, out of focus, glitch, hands, feet, bad anatomy`
