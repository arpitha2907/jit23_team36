import io

from PIL import Image

from app.services.render_preview import render_layout_image


def test_layout_preview_preserves_requested_dimensions_and_png_format():
    base = io.BytesIO()
    Image.new("RGB", (120, 80), (238, 236, 226)).save(base, format="JPEG", quality=92)

    rendered = render_layout_image(
        120,
        80,
        [
            {
                "text": "Corrected",
                "left": 20,
                "top": 25,
                "width": 70,
                "height": 18,
                "conf": 85,
            }
        ],
        base_image_bytes=base.getvalue(),
    )

    output = Image.open(io.BytesIO(rendered))
    assert output.format == "PNG"
    assert output.size == (120, 80)
