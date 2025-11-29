class MotifLibrary:
    """
    Library of common logical motifs for hypothesis generation.
    """
    def __init__(self):
        self.motifs = {}

    def get_motif(self, name):
        return self.motifs.get(name)
