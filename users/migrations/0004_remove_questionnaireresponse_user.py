# Generated by Django 4.2.18 on 2025-03-02 14:53

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0003_alter_userregistration_is_active_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='questionnaireresponse',
            name='user',
        ),
    ]
