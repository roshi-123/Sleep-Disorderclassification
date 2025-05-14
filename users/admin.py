from django.contrib import admin
from users.models import QuestionnaireResponse, UserRegistration
# Register your models here.

class UserRegistrationAdmin(admin.ModelAdmin):
    list_display = ('id','name', 'email', 'phonenumber', 'address', 'username', 'password',)

admin.site.register(UserRegistration,UserRegistrationAdmin)    


admin.site.register(QuestionnaireResponse)