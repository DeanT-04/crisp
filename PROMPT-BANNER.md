# Banner image prompt

Paste into an image generator (aspect ratio 3:1, target ~1280×400px).

## Palette

| Name | Hex | Used for |
|---|---|---|
| Blueprint navy | `#0E1A2B` | Banner background, badge `labelColor` |
| Crisp gold | `#F6B93B` | The crisp mascot, Python badge |
| Ketchup red | `#E8473F` | Accents (grin, crumbs), status badge |
| Vinegar mint | `#3DDC97` | Highlights, license badge |
| Blueprint cyan | `#5BC0EB` | Technical-drawing lines, "AI: none" badge |
| Paper cream | `#FFF6E5` | Title lettering, glints |

## Prompt

> A goofy, cheerful flat-vector cartoon illustration, wide 3:1 banner. Centre-left
> stands a single wavy golden potato crisp brought to life: big toothy grin,
> googly cartoon eyes, tiny round engineer's spectacles, holding an oversized
> magnifying glass like a detective. It stands on an unrolled engineering blueprint
> that runs the full width of the banner. To the left of the crisp, the blueprint is
> chunky and pixelated: a blurry drawing of a bolt and a flange made of big blocky
> squares. The crisp is mid-crunch, munching one large square pixel like a cookie,
> with small square pixel-crumbs flying off. Everything seen through the magnifying
> glass and to the right of it is razor-sharp: clean cyan technical-drawing lines, a
> perfect circle with a diameter symbol, dimension arrows with neat tick marks (no
> numbers). Palette: blueprint navy (#0E1A2B) background with a faint grid, crisp
> gold (#F6B93B) for the mascot, ketchup red (#E8473F) and vinegar mint (#3DDC97)
> accents, blueprint cyan (#5BC0EB) drawing lines, paper cream (#FFF6E5) lettering.
> Typography: the single word "crisp" in chunky, rounded, bold lowercase letters,
> paper cream with a thick blueprint-navy outline and a small drop shadow, placed in
> the right third with a slight playful bounce; the left edge of the "c" is
> pixelated and the rest of the word is perfectly sharp. Style notes: thick,
> confident outlines and flat colour fills, with one soft cel-shade on the crisp.
> Bright Saturday-morning-cartoon mood, but the drawing lines stay precise and
> engineered. The joke is a silly snack fixing a very serious blueprint.

**Typography decision:** the name is rendered in-image. If the generator garbles
it, generate without text and add "crisp" in post using a rounded bold face
(e.g. Fredoka Bold or Baloo 2 ExtraBold) in paper cream with a navy outline.

**Negative prompt:** realistic photo of chips, crisp packets or brand logos,
text artifacts, garbled letters, numbers or labels on the dimension lines, extra
limbs, more than one mascot, watermark, signature, busy or cluttered composition,
off-palette colours, muddy gradients, 3D render, stock-photo look, low contrast.

**Save to:** `docs/assets/banner.png`
