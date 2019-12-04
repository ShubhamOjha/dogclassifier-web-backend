from rest_framework import serializers
from .models import ProcessedImage
from .helper import DogClassifier

class ProcessedImageSerializer(serializers.ModelSerializer):
    
    dog_breed = serializers.SerializerMethodField('get_dog_breed')
      
    def get_dog_breed(self, obj):
        dc = DogClassifier(obj.file.path)
        return dc.dog_classifer()
    
    class Meta:
        model = ProcessedImage
        fields = ('id', 'file', 'dog_breed')