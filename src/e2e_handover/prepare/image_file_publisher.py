""" Node that publishes images. Recursively descends given directory to find folders that match expression and publishes all images in that dir. """

class ImagePublisher():
    def __init__(self):
        pass

def main():
    image_publisher = ImagePublisher()
    image_publisher.run()

if __name__ == "__main__":
    main()