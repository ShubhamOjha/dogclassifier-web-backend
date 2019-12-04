from django.shortcuts import render
from rest_framework.parsers import FileUploadParser
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from .serializers import ProcessedImageSerializer

class ProcessedImageView(APIView):

    parser_class = (FileUploadParser,)

    def post(self, request, *args, **kwargs):
        image_serializer = ProcessedImageSerializer(data=request.data)

        if image_serializer.is_valid():
            image_serializer.save()
            return Response(image_serializer.data, status=status.HTTP_200_OK)
        else:
            return Response(image_serializer.data, status=status.HTTP_400_BAD_REQUEST)
