import io
import base64

from PIL import Image

class Tools:
    def __init__(self):
        pass

    @staticmethod
    def bytes_to_image(data: bytes) -> Image.Image:
        """Конвертирует байты, полученные от клиента, в объект PIL.Image."""
        try:
            image = Image.open(io.BytesIO(data))
            image = image.convert("RGB")
            return image
        except Exception as exc:  
            raise ValueError(f"Не удалось прочитать изображение: {exc}") from exc

    @staticmethod
    def image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
        """Кодирует PIL.Image в base64-строку для отправки в JSON-ответе."""
        buffer = io.BytesIO()
        image.save(buffer, format=fmt)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/{fmt.lower()};base64,{encoded}"

    def base64_to_image(self, data: str) -> Image.Image:
        """Декодирует base64-строку (в т.ч. с data:URL префиксом) обратно в PIL.Image."""
        if "," in data and data.strip().startswith("data:"):
            data = data.split(",", 1)[1]
        raw = base64.b64decode(data)
        return self.bytes_to_image(raw)