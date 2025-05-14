from django.contrib import admin
from django.urls import path
from users import views as UserView
from admins import views
from users.utility import training_model
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', UserView.Index, name='index'),

    path('adminLogin/', views.AdminLogin, name='adminLogin'),
    path('adminlogout/', views.AdminLogout, name='adminlogout'),
    path('adminhome/', views.AdminHome, name='adminhome'),

    path('activatedusers/', views.ActivatedUsers, name='activateduser'),
    path('useractivate/<int:pk>', views.UserActivate, name='useractivate'),
    path('Blockuser/<int:pk>', views.BlockUser, name='blockuser'),

    path('modelEvaluation/', training_model.Model_Evaluation_View, name='modeleval'),
    path('modelprediction/', training_model.ModelPrediction, name='modelprediction'),

    path('userLogin/', UserView.UserLoginPage, name='userLogin'),
    path('userlogout/', UserView.UserLogout, name='userLogout'),
    path('UserRegister/', UserView.UserRegistrationPage, name='userregister'),

    path('UserHome/', UserView.UserHomePage, name='UserHome'),
    path('datasetview/', UserView.DataSetView, name='datasetview'),
    path('matrices/', UserView.ModelMatrices, name='modelmatrices'),

    path('about/', UserView.about, name='about'),
    path('contactus/', UserView.contactus, name='contactus'),

    # Corrected questionnaire URLs
    path('questionnaire_form/', UserView.questionnaire_form, name='questionnaire_form'),  # Display form
    path('save_questionnaire', UserView.save_questionnaire, name='save_questionnaire'),  # Handle form submission

    # path('prediction/', modelprediction, name='modelprediction'),
] 

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
