import os


IGNORE_INDEX = -100


def get_cached_image_count(cache_dir):
    """
    Return the number of cached images.
    """
    images_dir = os.path.join(cache_dir, "images")
    if not os.path.isdir(images_dir):
        return 0
    count = 0
    while os.path.isfile(os.path.join(images_dir, f"{count}.png")):
        count += 1
    return count


