# Generated by Django 5.1.4 on 2024-12-31 13:09

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='MonModele',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('nom', models.CharField(max_length=255)),
                ('valeur1', models.FloatField()),
                ('valeur2', models.FloatField()),
                ('valeur3', models.FloatField()),
            ],
        ),
    ]
