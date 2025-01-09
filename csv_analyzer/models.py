from django.db import models

class MonModele(models.Model):
    nom = models.CharField(max_length=255)
    valeur1 = models.FloatField()
    valeur2 = models.FloatField()
    valeur3 = models.FloatField()
    # ... autres champs

    def __str__(self):
        return self.nom