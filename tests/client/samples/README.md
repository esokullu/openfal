# Reference fixtures

`sample_portrait_photo.png` is the default photographic identity fixture for manual
quality checks. It depicts a fictional adult and was generated with the built-in
imagegen tool on 2026-11-17. Look for stable face shape, hair, freckles and age,
natural skin texture, and the requested change in clothing, light or medium.
An HTTP success and a decodable PNG alone do not establish those properties.

`sample_portrait.png` remains a deliberately simple drawing for smoke tests.
Preserving its cartoon face is correct identity conditioning, not evidence that
the model cannot produce photographic faces. Set `OPENFAL_PORTRAIT=sample_portrait.png`
to exercise this fixture with the Node clients.

`sample_style_photo.png` is the default photographic lighting reference, also
generated with the built-in imagegen tool on 2026-11-17. Its plain facades avoid
accidental text or geometric shapes being transferred into the edit.
`sample_style_ref.png` and `sample_street_scene.png` remain stylized smoke-test
references. Select the old lighting fixture with `OPENFAL_STYLE=sample_style_ref.png`.

`sample_street_photo.png` is the default I2V/reference scene. It was generated
locally by the corrected MiniMax-H3-Image runner on 2026-11-17 with seed `946813846`,
at 1344x768, using the installed INT8 ConvRot checkpoint and its six-step PDD schedule.
Prompt: "A photorealistic wide shot of an orange tabby cat chasing a white poodle
across an Istanbul cobblestone alley at sunset, Bosphorus in background".
It is a conditioning input, not an expected-output golden image.
Set `OPENFAL_STREET=sample_street_scene.png` to use the geometric smoke-test scene.

Photographic fixture generation prompt (built-in imagegen, no input image):

> Use case: photorealistic-natural. Asset type: square reference portrait for an
> open source image editing model test suite. Create one fictional adult person,
> age about 35, shoulder-up frontal portrait, calm neutral expression, eyes visible
> and looking toward camera, slightly asymmetrical human face, short wavy dark
> hair, brown eyes, subtle freckles, natural skin pores, light stubble, plain
> slate-blue cotton crewneck shirt, warm neutral light-grey studio background.
> Soft even daylight, realistic documentary photograph, 85mm lens, sharp focus
> throughout face, enough shoulders visible for clothing edits. Keep neutral
> lighting and no dramatic grade so later model tests can relight and restyle it.
> No celebrities or recognizable real person, no text, no jewelry, no eyewear,
> no watermarks. Single continuous photograph, 1024x1024 square.

Lighting fixture generation prompt (built-in imagegen, no input image):

> Use case: photorealistic-natural. Asset type: square lighting reference fixture
> for an open source image editing test suite. A realistic editorial photograph
> of an empty rainy side street in Istanbul at night, traditional late Ottoman
> stone and timber facades, narrow cobblestone street with wet reflections, subtle
> cyan and magenta architectural accent lighting on plain walls, warm small lamps
> in doorways. Eye-level natural perspective, believable restrained cinematic
> atmosphere, detailed real masonry and rain. No people, no faces, no signs,
> no lettering, no neon text, no logos, no light rings, no sci-fi laser grid,
> no illustration. This image will guide lighting and colors only, not identity
> or props. Single continuous square photograph.
