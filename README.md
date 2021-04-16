# discountofx
Simple OFX video filter plugins

This is a set of CUDA-accelerated OpenFX video plugins for use in programs like Davinci Resolve. They were tested with Resolve 17. The non-CUDA software fallback has not been tested, but should work.

There are also several false color LUTs to aid in color grading.

## OFX Filters

### Denoise

The denoise filter is a non-local-means denoise filter. The parameters are:
* Amount: how strong the denoising effect is
* Luminance: what percent of the denoising should apply to luminance
* Color: what percent of the denoising should apply to color information
* Width: how wide the filter searches for values when applying denoising. Higher values result in slower rendering.

### Discount Denoise

This is the same non-local-means denoising, but with optimizations to improve rendering time. In particular, the width is fixed, and the denoising only applies to the color channel. The only parameter is:
* Amount: how strong the denoising effect is

### Desharp

This filter applies a very localized blur, meant to counteract footage that has been digitally sharpened. For example, some drones bake a lot of digital sharpening into their video files, creating artifacts. This filter attempts to reverse that sharpening filter, but may do a better or worse job depending on the sharepening algorithm originally used. The parameters are:
* Desharpen: the amount of blurring
* Sharpen: if you want to apply ugly digital sharpening instead

### Gamma adjust

This filter applies an exact gamma curve to your footage, useful if you need to correct gamma by an exact amount instead of adjusting gamma with color grading tools. The equation is simply x^g where x is your footage and g is the parameter:
* Gamma: g in the equation x^g

### Soft Saturate

This filter allows adjustment of color saturation on a curve. If you're using Resolve, this plugin isn't useful because it's exactly the same as the "Color Boost" and "Saturation" controls. The parameters are:
* Exponent: this is called "Color Boost" by Resolve. Values less than 1 will push unsaturated colors to be more saturated, without affecting fully saturated colors as much. Values greater than 1 will push unsaturated colors to be less saturated, without affecting fully saturated colors as much.
* Gain: this is the typical "saturation" control available in color grading

### Color Gain Controls

This filter is a 3x3 color multiplication matrix for color rotations, etc. Described another way, you are able to precisely specify what the output Red, Green, and Blue channels should be by composing inputs from each of the R, G, and B channel inputs. This can be useful when trying to color match cameras, as part of film emulation, etc.

For example, if you want the output red channel to be composed mostly of red but also have a little of whatever was in the input green and blue channels, you'd set R scale R to something close to one, and R scale G (and B) to some low number. You could then do the same with the other channels.

The Preserve Luminance slider, when set to 1, normalizes the output to maintain the same luminance as the input.

## LUTs

### GreyChecker

This LUT outputs green for areas of the image that are desaturated. This can be useful to help double check your white balance (white or grey things should show up green). The LUT fades from green to red as the color gets more saturated, and then goes to greyscale for anything that's decently saturated.

### LumaZones

This LUT outputs a color for each of the Ansel Adams luminance zones, from black to purple, to blue, all the way to red.

### Lutgen

This poorly named LUT outputs a RED-like false color mapping (dark skin tones are green, light skin tones are pink, teal is for shadows, overly bright is red, and overly dark is purple. Everything else is greyscale.

### SaturationCheck

This LUT checks for close-to-clipped luminance or color gamut problems and outputs red for those areas. Everything else is greyscale.

### SkinCheck

Did you know everyone's skintone is actually in a narrow hue range, regardless of skin color? Skin color changes luminance, but not hue. Anyway, to make sure you don't have an unexpected tint to your skin tones, this LUT amplifies the narrow hue range of typical skin tones into very large hue differences. If your skin tones are skewing toward a very slightly greenish hue, the LUT will show that skin as very green. Likewise with a slight magenta shift. Everything else is coloured black. The amplification of hues is so strong that you may find that a particular tint (magenta) always looks correct for a person, which is fine, that may just be that person's natural tint, and will at least help you get the tint consistent shot-to-shot.
