from django.shortcuts import render, redirect
from django.contrib.auth import logout
from django.contrib import messages
from . models import UserRegistration
from django.conf import settings
import os
import pandas as pd

def Index(request):
    return render(request, 'index.html')

def UserRegistrationPage(request):
    if request.method =='POST':
        name = request.POST['name']
        email = request.POST['email']
        phonenumber = request.POST['phone']
        address = request.POST['address']
        username = request.POST['username']
        password = request.POST['pswd']
        userdetails = UserRegistration.objects.create(
            name = name,
            email = email,
            phonenumber = phonenumber,
            address = address,
            username = username,
            password = password,
        )
        userdetails.save()
        messages.success(request, 'Registration Successful')
        print('Registration Success............')
    return render(request, 'userregisterpage.html')  
  

def UserLoginPage(request):
    if request.method == 'POST':
        UserName = request.POST['username']
        Password = request.POST['pswd']
        try:
            user = UserRegistration.objects.get(username=UserName, password=Password)
            if user.is_active:
                request.session['id']=user.id
                request.session['username'] = user.username
                request.session['email'] = user.email   
                request.session['name'] = user.name

                return redirect('UserHome')
            else:
                messages.warning(request, 'Your account is not active. Please Activate through Admin.')
                return redirect('userLogin')
        except UserRegistration.DoesNotExist:
            messages.warning(request, 'Check your login details.')
            return redirect('userLogin')
    return render(request, 'userloginpage.html')


def UserHomePage(request):
    name = request.session.get('name')
    email = request.session.get('email')
    username = request.session.get('username')
    return render(request, 'users/userbase.html', {'name':name, 'email':email,'username':username})


def DataSetView(request):
    path = os.path.join(settings.MEDIA_ROOT, 'data', 'sleep_health_and_lifestyle_dataset.csv')
    data = pd.read_csv(path)
    return render(request, 'users/datasetView.html', {'data': data.to_html(classes='table table-striped table-bordered')})


def ModelMatrices(request):
    return render(request, 'analysis/matrices.html')

def UserLogout(request):
    logout(request)
    return redirect('userLogin')

def about(request):
    return render(request, 'about.html')

def contactus(request):
    return render(request, 'contactus.html')

def questionnaire_form(request):
    return render(request, "analysis/questionnaire_form.html")

from django.shortcuts import render, redirect
from django.contrib import messages
from .models import QuestionnaireResponse

def save_questionnaire(request):
    print("HIIIIIIIIIIIIii")
    if request.method == 'POST':
        id = request.POST.get('id')
        wake_freshness = request.POST.get('wake_freshness')
        wake_frequency = request.POST.get('wake_frequency')
        stress_level = request.POST.get('stress_level')
        stress_management = request.POST.get('stress_management')
        exercise_frequency = request.POST.get('exercise_frequency')
        weight_change = request.POST.get('weight_change')
        device_usage = request.POST.get('device_usage')
        caffeine_alcohol = request.POST.get('caffeine_alcohol')
        concentration = request.POST.get('concentration')
        print(stress_level)
        try:
            userqa = QuestionnaireResponse.objects.get(id=id)  # Check if record exists
            if userqa:
                userqa.wake_frequency=wake_frequency
                userqa.wake_freshness=wake_freshness
                userqa.stress_level=stress_level
                userqa.stress_management=stress_management
                userqa.exercise_frequency=exercise_frequency
                userqa.weight_change=weight_change
                userqa.device_usage=device_usage
                userqa.caffeine_alcohol=caffeine_alcohol
                userqa.concentration=concentration
                userqa.save()
        except QuestionnaireResponse.DoesNotExist:    
            qaw = QuestionnaireResponse(
                id=id,
                wake_freshness=wake_freshness,
                wake_frequency=wake_frequency,
                stress_level=stress_level,
                stress_management=stress_management,
                exercise_frequency=exercise_frequency,
                weight_change=weight_change,
                device_usage=device_usage,
                caffeine_alcohol=caffeine_alcohol,
                concentration=concentration
            )
            qaw.save()
       
            messages.success(request, "🎉 Thank you! Your responses have been saved.")
            return render(request, 'analysis/predictionpage.html')

        
        messages.success(request, "🎉 Thank you! Your responses have been saved.")
        return render(request, 'analysis/predictionpage.html')


    messages.success(request, "Error occured!")
    return render(request, 'analysis/questionnaire_form.html')
    


from django.contrib import messages
from django.shortcuts import redirect, render
from .models import QuestionnaireResponse

# def modelprediction(request):
#     id=request.GET['id']
#     # Fetch the latest questionnaire response from any user
    

#     if request.method == 'POST':
#         age = request.POST.get('age')
#         gender = request.POST.get('gender')
#         sleep = request.POST.get('sleep')
#         stress = request.POST.get('stress')

#         # **Validation Logic**
#         errors = []

#         if user_questionnaire.stress_level == "always" and int(stress) < 8:
#             errors.append("Your questionnaire indicates extreme stress, but you entered a low stress level.")

#         if user_questionnaire.wake_freshness == "no" and int(sleep) > 8:
#             errors.append("Your questionnaire suggests you wake up tired, but your sleep duration is too high.")

#         if user_questionnaire.exercise_frequency in ["rarely", "1-2"] and int(sleep) > 9:
#             errors.append("You reported low exercise levels, which might not align with high sleep duration.")

#         if user_questionnaire.caffeine_alcohol == "yes" and int(sleep) > 7:
#             errors.append("Your caffeine or alcohol intake may impact your sleep duration.")

#         if errors:
#             for error in errors:
#                 messages.error(request, error)
#             return redirect('predictionpage')

#         pred = "Prediction Result Here"
#         return render(request, 'analysis/predictionpage.html', {'pred': pred})

#     return render(request, 'modelprediction')



